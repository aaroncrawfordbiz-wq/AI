"""
A GPT-style transformer, from scratch in numpy.

This is the same architecture family as ChatGPT/Claude — token embeddings,
stacked self-attention blocks, and a next-token prediction head — in miniature,
with every forward pass AND every gradient written out by hand so there is no
magic anywhere.

Why attention makes it smarter than the old MLP model:
  The MLP saw a fixed window of the last 8 characters, all mashed together.
  ATTENTION lets every position look back at every earlier position and decide,
  with learned weights, WHICH previous characters matter right now. That is the
  breakthrough ("Attention Is All You Need", 2017) behind modern AI.

Layout of one forward pass:
  ids -> token embedding + position embedding
      -> [LayerNorm -> Multi-head causal self-attention -> add back (residual)
          LayerNorm -> MLP -> add back (residual)]  x n_layer
      -> LayerNorm -> Linear -> logits (a score for every character in the vocab)

Every class below has a forward() and a matching backward() that applies the
chain rule for exactly the math its forward() did.
"""

import json
import os

import numpy as np

rng = np.random.default_rng(1234)


class Config:
    """Hyperparameters — the knobs that set the model's size and shape."""

    def __init__(self, vocab_size, context=64, n_layer=3, n_head=4, n_emb=96,
                 dtype="float32"):
        self.vocab_size = vocab_size
        self.context = context      # how many characters the model can look back
        self.n_layer = n_layer      # how many transformer blocks are stacked
        self.n_head = n_head        # parallel attention "heads" per block
        self.n_emb = n_emb          # width of every internal vector
        self.dtype = dtype

    def to_dict(self):
        return dict(vocab_size=self.vocab_size, context=self.context,
                    n_layer=self.n_layer, n_head=self.n_head,
                    n_emb=self.n_emb, dtype=self.dtype)


class Linear:
    """y = x @ W + b — the workhorse layer, same as step 1's neuron but wide."""

    def __init__(self, n_in, n_out, cfg, std=0.02):
        self.W = rng.normal(0, std, (n_in, n_out)).astype(cfg.dtype)
        self.b = np.zeros(n_out, dtype=cfg.dtype)
        self.dW = self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        d2 = d.reshape(-1, d.shape[-1])
        self.dW = x2.T @ d2
        self.db = d2.sum(0)
        return d @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class LayerNorm:
    """Rescales each vector to mean 0 / spread 1, then applies a learned
    scale and shift. Keeps signals stable so deep stacks can train."""

    def __init__(self, n, cfg):
        self.g = np.ones(n, dtype=cfg.dtype)
        self.b = np.zeros(n, dtype=cfg.dtype)
        self.dg = self.db = None

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        self.std = np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        self.xhat = (x - mu) / self.std
        return self.g * self.xhat + self.b

    def backward(self, d):
        n = d.shape[-1]
        self.dg = (d * self.xhat).reshape(-1, n).sum(0)
        self.db = d.reshape(-1, n).sum(0)
        dxhat = d * self.g
        # chain rule through the mean and variance (the classic layernorm grad)
        return (dxhat
                - dxhat.mean(-1, keepdims=True)
                - self.xhat * (dxhat * self.xhat).mean(-1, keepdims=True)
                ) / self.std

    def params_and_grads(self):
        return [(self.g, self.dg), (self.b, self.db)]


class CausalSelfAttention:
    """The heart of a transformer.

    Each position emits a Query ("what am I looking for?"), a Key ("what do I
    contain?") and a Value ("what do I pass along?"). Position i attends to
    position j with weight softmax(q_i · k_j) — but only for j <= i ("causal":
    you may not peek at the future you're trying to predict). Multiple heads
    do this in parallel so different heads can track different things
    (one may track spaces, another recent letters, another quotes...).
    """

    def __init__(self, cfg):
        E = cfg.n_emb
        self.n_head = cfg.n_head
        self.qkv = Linear(E, 3 * E, cfg)     # makes q, k, v all at once
        self.proj = Linear(E, E, cfg)        # mixes the heads back together
        # causal mask: True where attention is allowed (lower triangle)
        self.mask = np.tril(np.ones((cfg.context, cfg.context), dtype=bool))

    def forward(self, x):
        B, T, E = x.shape
        H = self.n_head
        hs = E // H                          # size of each head
        qkv = self.qkv.forward(x)
        q, k, v = np.split(qkv, 3, axis=-1)
        # reshape to (batch, head, position, head_size)
        q = q.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hs).transpose(0, 2, 1, 3)

        # hs ** 0.5 (a plain Python float) rather than np.sqrt(hs): numpy 2
        # would silently promote every float32 array downstream to float64
        att = q @ k.transpose(0, 1, 3, 2) / (hs ** 0.5)   # (B,H,T,T) scores
        att = np.where(self.mask[:T, :T], att, -1e9)      # block the future
        att = att - att.max(-1, keepdims=True)            # numerical safety
        e = np.exp(att)
        self.S = e / e.sum(-1, keepdims=True)             # attention weights
        y = self.S @ v                                    # weighted mix of values

        self.q, self.k, self.v = q, k, v
        out = y.transpose(0, 2, 1, 3).reshape(B, T, E)
        return self.proj.forward(out)

    def backward(self, d):
        B, T, E = self.q.shape[0], self.q.shape[2], self.n_head * self.q.shape[3]
        H, hs = self.n_head, self.q.shape[3]
        d = self.proj.backward(d)
        dy = d.reshape(B, T, H, hs).transpose(0, 2, 1, 3)

        dS = dy @ self.v.transpose(0, 1, 3, 2)
        dv = self.S.transpose(0, 1, 3, 2) @ dy
        # softmax backward (masked spots have S=0, so their gradient is 0)
        datt = self.S * (dS - (dS * self.S).sum(-1, keepdims=True))
        datt = datt / (hs ** 0.5)
        dq = datt @ self.k
        dk = datt.transpose(0, 1, 3, 2) @ self.q

        merge = lambda a: a.transpose(0, 2, 1, 3).reshape(B, T, E)
        dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
        return self.qkv.backward(dqkv)

    def params_and_grads(self):
        return self.qkv.params_and_grads() + self.proj.params_and_grads()


class MLP:
    """Two linear layers with a ReLU between — where the model 'thinks' about
    what attention gathered. 4x wider inside, like real GPTs."""

    def __init__(self, cfg):
        self.fc = Linear(cfg.n_emb, 4 * cfg.n_emb, cfg)
        self.out = Linear(4 * cfg.n_emb, cfg.n_emb, cfg)

    def forward(self, x):
        self.h = np.maximum(0, self.fc.forward(x))
        return self.out.forward(self.h)

    def backward(self, d):
        d = self.out.backward(d)
        d = d * (self.h > 0)
        return self.fc.backward(d)

    def params_and_grads(self):
        return self.fc.params_and_grads() + self.out.params_and_grads()


class Block:
    """One transformer block. The 'x +' parts are residual connections:
    each sub-layer only ADDS a correction to the signal, which is what lets
    gradients flow cleanly through deep stacks."""

    def __init__(self, cfg):
        self.ln1 = LayerNorm(cfg.n_emb, cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_emb, cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x

    def backward(self, d):
        d = d + self.ln2.backward(self.mlp.backward(d))
        d = d + self.ln1.backward(self.attn.backward(d))
        return d

    def params_and_grads(self):
        return (self.ln1.params_and_grads() + self.attn.params_and_grads()
                + self.ln2.params_and_grads() + self.mlp.params_and_grads())


class GPT:
    def __init__(self, cfg):
        self.cfg = cfg
        E = cfg.n_emb
        self.tok_emb = rng.normal(0, 0.02, (cfg.vocab_size, E)).astype(cfg.dtype)
        self.pos_emb = rng.normal(0, 0.01, (cfg.context, E)).astype(cfg.dtype)
        self.dtok_emb = self.dpos_emb = None
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(E, cfg)
        self.head = Linear(E, cfg.vocab_size, cfg)

    def num_params(self):
        return sum(p.size for p, _ in self.params_and_grads())

    def forward(self, idx, targets=None):
        """idx: (batch, time) integer character ids. Returns logits and,
        if targets given, the cross-entropy loss."""
        B, T = idx.shape
        self.idx = idx
        x = self.tok_emb[idx] + self.pos_emb[:T]   # what + where
        for blk in self.blocks:
            x = blk.forward(x)
        x = self.ln_f.forward(x)
        logits = self.head.forward(x)

        if targets is None:
            return logits, None
        self.targets = targets
        z = logits - logits.max(-1, keepdims=True)
        e = np.exp(z)
        se = e.sum(-1, keepdims=True)
        self.probs = e / se
        logp = z - np.log(se)          # exact log-softmax — matches backward
        flat = logp.reshape(B * T, -1)
        loss = -flat[np.arange(B * T), targets.reshape(-1)].mean()
        return logits, loss

    def backward(self):
        """Chain rule from the loss back to every parameter."""
        B, T = self.idx.shape
        dlogits = self.probs.copy()
        flat = dlogits.reshape(B * T, -1)
        flat[np.arange(B * T), self.targets.reshape(-1)] -= 1
        dlogits /= B * T

        d = self.head.backward(dlogits)
        d = self.ln_f.backward(d)
        for blk in reversed(self.blocks):
            d = blk.backward(d)
        # gradients reach the embeddings too — the model learns what each
        # character MEANS and what each position means
        self.dtok_emb = np.zeros_like(self.tok_emb)
        np.add.at(self.dtok_emb, self.idx, d)
        self.dpos_emb = np.zeros_like(self.pos_emb)
        self.dpos_emb[:T] = d.sum(0)

    def params_and_grads(self):
        pairs = [(self.tok_emb, self.dtok_emb), (self.pos_emb, self.dpos_emb)]
        for blk in self.blocks:
            pairs += blk.params_and_grads()
        return pairs + self.ln_f.params_and_grads() + self.head.params_and_grads()

    def generate(self, ids, n_tokens, temperature=0.8, top_k=40, stream=None):
        """Autoregression — the GPT loop: predict, sample, append, repeat."""
        ids = list(ids)
        out = []
        for _ in range(n_tokens):
            ctx = np.array([ids[-self.cfg.context:]])
            logits, _ = self.forward(ctx)
            logits = logits[0, -1] / max(temperature, 1e-6)
            if top_k:                       # only consider the k best choices
                cutoff = np.sort(logits)[-min(top_k, len(logits))]
                logits = np.where(logits < cutoff, -np.inf, logits)
            z = logits - logits.max()
            p = np.exp(z)
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
            ids.append(nxt)
            out.append(nxt)
            if stream:
                stream(nxt)
        return out


class Adam:
    """A smarter version of 'nudge by the gradient': it keeps a running memory
    of each parameter's gradient (momentum) and gradient size (so every weight
    gets a step scaled to its own history). Trains far faster than plain SGD."""

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m, self.v, self.t = None, None, 0

    def step(self, pairs, lr):
        if self.m is None:
            self.m = [np.zeros_like(p) for p, _ in pairs]
            self.v = [np.zeros_like(p) for p, _ in pairs]
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for i, (p, g) in enumerate(pairs):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mhat = self.m[i] / (1 - b1 ** self.t)
            vhat = self.v[i] / (1 - b2 ** self.t)
            p -= lr * mhat / (np.sqrt(vhat) + self.eps)


# ---------------- checkpoints: save a trained brain, load it later ----------

def save_checkpoint(path, model, chars):
    meta = json.dumps({"config": model.cfg.to_dict(), "vocab": chars})
    arrays = {f"p{i}": p for i, (p, _) in enumerate(model.params_and_grads())}
    # write to a temp file then swap it in, so an interrupt mid-save can
    # never destroy the previous good checkpoint
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=meta, **arrays)
    os.replace(tmp, path)


def load_checkpoint(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    cfg = Config(**meta["config"])
    model = GPT(cfg)
    for i, (p, _) in enumerate(model.params_and_grads()):
        p[...] = z[f"p{i}"]
    return model, meta["vocab"]
