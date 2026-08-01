"""
Train the AI. It saves its brain to a checkpoint file, so you train ONCE and
then generate from it forever (no more relearning every run).

Quick start:
  python train.py --dataset shakespeare            # learns English (~5 min)
  python train.py --dataset code                   # learns Python  (~5 min)
  python train.py --data data/mybook.txt --out checkpoints/mybook

Make it smarter (bigger model, longer training — slower but better):
  python train.py --dataset shakespeare --layers 4 --emb 128 --context 128 --steps 8000

Then talk to it:
  python generate.py --model checkpoints/shakespeare.npz --prompt "ROMEO:"
"""

import argparse
import os
import time

import numpy as np

from download_data import get_dataset
from model import GPT, Adam, Config, save_checkpoint, load_checkpoint
from tokenizer import CharTokenizer

rng = np.random.default_rng(0)


def get_batch(ids, batch, context):
    """Grab `batch` random snippets. x is the text, y is the same text shifted
    one character left — at every position the model predicts the NEXT char."""
    starts = rng.integers(0, len(ids) - context - 1, batch)
    x = np.stack([ids[s:s + context] for s in starts])
    y = np.stack([ids[s + 1:s + context + 1] for s in starts])
    return x, y


def val_loss(model, ids, batch, context, rounds=8):
    total = 0.0
    for _ in range(rounds):
        x, y = get_batch(ids, batch, context)
        _, loss = model.forward(x, y)
        total += loss
    return total / rounds


def lr_at(step, steps, lr_max):
    """Warm up, then cosine-decay — starts gentle, ends gentle. Standard."""
    warmup = min(200, steps // 10)
    if step < warmup:
        return lr_max * (step + 1) / warmup
    frac = (step - warmup) / max(1, steps - warmup)
    return 0.1 * lr_max + 0.45 * lr_max * (1 + np.cos(np.pi * frac))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="shakespeare or code (auto-downloads)")
    p.add_argument("--data", help="path to any .txt file to learn from")
    p.add_argument("--out", help="checkpoint path (default checkpoints/<name>)")
    p.add_argument("--resume", help="checkpoint to continue training from")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--emb", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-3)
    args = p.parse_args()

    # ---- data ----
    if args.data:
        path, name = args.data, os.path.splitext(os.path.basename(args.data))[0]
    elif args.dataset:
        path, name = get_dataset(args.dataset), args.dataset
    else:
        raise SystemExit("pick data:  --dataset shakespeare | --dataset code | --data file.txt")
    text = open(path, encoding="utf-8", errors="ignore").read()

    # ---- model (fresh, or resumed from a saved brain) ----
    if args.resume:
        model, chars = load_checkpoint(args.resume)
        tok = CharTokenizer(chars)
        print(f"resumed {args.resume}")
    else:
        tok = CharTokenizer.from_text(text)
        cfg = Config(vocab_size=tok.vocab_size, context=args.context,
                     n_layer=args.layers, n_head=args.heads, n_emb=args.emb)
        model = GPT(cfg)

    ids = tok.encode(text)
    split = int(0.95 * len(ids))
    train_ids, valid_ids = ids[:split], ids[split:]
    ctx = model.cfg.context

    out = args.out or os.path.join("checkpoints", name)
    if not out.endswith(".npz"):
        out += ".npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"data: {len(text):,} chars, vocab {tok.vocab_size} | "
          f"model: {model.num_params():,} parameters | "
          f"random-guess loss would be {np.log(tok.vocab_size):.2f}")

    # ---- the loop: forward -> loss -> backward -> Adam nudge ----
    opt = Adam()
    t0 = time.time()
    for step in range(args.steps + 1):
        x, y = get_batch(train_ids, args.batch, ctx)
        _, loss = model.forward(x, y)
        model.backward()
        opt.step(model.params_and_grads(), lr_at(step, args.steps, args.lr))

        if step % 100 == 0:
            speed = (step + 1) / (time.time() - t0)
            print(f"step {step:5d}/{args.steps}   train loss {loss:.3f}   "
                  f"({speed:.1f} steps/s)")
        if step and step % 500 == 0:
            vl = val_loss(model, valid_ids, args.batch, ctx)
            print(f"          validation loss {vl:.3f}  (loss on text it has "
                  f"never seen — the honest score)")
            save_checkpoint(out, model, tok.chars)
            sample = model.generate(tok.encode(text[:ctx]), 150)
            print(f"          sample: {tok.decode(sample)!r}\n")

    save_checkpoint(out, model, tok.chars)
    print(f"\nbrain saved to {out}")
    print(f"try it:  python generate.py --model {out} --prompt \"{text[:20]}\"")


if __name__ == "__main__":
    main()
