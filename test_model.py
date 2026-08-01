"""
Proof the math is right.

Test 1 — gradient check: for a tiny model, compare our hand-written backprop
against brute-force numerical derivatives (nudge each weight by a hair, watch
the loss move). If backprop is wrong anywhere, this catches it.

Test 2 — learning check: train briefly on a tiny repeating text and verify the
loss collapses (the model memorizes it) and a checkpoint round-trips exactly.

  python test_model.py
"""

import os
import tempfile

import numpy as np

from model import GPT, Adam, Config, load_checkpoint, save_checkpoint
from tokenizer import CharTokenizer


def gradient_check():
    cfg = Config(vocab_size=11, context=6, n_layer=2, n_head=2, n_emb=8,
                 dtype="float64")           # float64: exact enough to compare
    model = GPT(cfg)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 11, (2, 6))
    y = rng.integers(0, 11, (2, 6))

    _, loss = model.forward(x, y)
    model.backward()
    pairs = model.params_and_grads()

    h = 1e-5
    worst = 0.0
    for pi, (p, g) in enumerate(pairs):
        flat_p, flat_g = p.reshape(-1), g.reshape(-1)
        # spot-check a handful of entries in every parameter tensor
        for j in rng.choice(flat_p.size, size=min(5, flat_p.size), replace=False):
            old = flat_p[j]
            flat_p[j] = old + h
            _, l1 = model.forward(x, y)
            flat_p[j] = old - h
            _, l2 = model.forward(x, y)
            flat_p[j] = old
            numeric = (l1 - l2) / (2 * h)
            denom = max(abs(numeric) + abs(flat_g[j]), 1e-8)
            rel = abs(numeric - flat_g[j]) / denom
            worst = max(worst, rel)
            assert rel < 1e-4, (
                f"GRADIENT MISMATCH in param {pi} at {j}: "
                f"backprop={flat_g[j]:.8f} numeric={numeric:.8f}")
    print(f"PASS gradient check — backprop matches calculus on every layer "
          f"(worst relative error {worst:.2e})")


def learning_check():
    text = "hello tiny world. " * 60
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    cfg = Config(vocab_size=tok.vocab_size, context=16, n_layer=2, n_head=2,
                 n_emb=32)
    net = GPT(cfg)
    opt = Adam()
    rng = np.random.default_rng(1)

    loss = None
    for step in range(400):
        starts = rng.integers(0, len(ids) - 17, 16)
        x = np.stack([ids[s:s + 16] for s in starts])
        y = np.stack([ids[s + 1:s + 17] for s in starts])
        _, loss = net.forward(x, y)
        net.backward()
        opt.step(net.params_and_grads(), 1e-2)
    assert loss < 0.5, f"model failed to learn, final loss {loss:.3f}"
    print(f"PASS learning check — loss fell to {loss:.3f} "
          f"(random guessing would be {np.log(tok.vocab_size):.2f})")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ck.npz")
        save_checkpoint(path, net, tok.chars)
        net2, chars2 = load_checkpoint(path)
        assert chars2 == tok.chars
        for (a, _), (b, _) in zip(net.params_and_grads(), net2.params_and_grads()):
            assert np.array_equal(a, b), "checkpoint changed the weights!"
    print("PASS checkpoint check — saved and reloaded brain is identical")

    out = net.generate(tok.encode("hello"), 24, temperature=0.3)
    print(f"sample from the overfit model: {'hello' + tok.decode(out)!r}")


if __name__ == "__main__":
    gradient_check()
    learning_check()
    print("\nall tests passed")
