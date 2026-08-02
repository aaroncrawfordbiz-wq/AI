"""
Train the AI. It saves its brain to a checkpoint file, so you train ONCE and
then generate from it forever (no more relearning every run).

Quick start:
  python train.py --dataset shakespeare        # learns English (~10-15 min)
  python train.py --dataset code               # learns Python  (~10-15 min)
  python train.py --data data/mybook.txt --out checkpoints/mybook

Make it smarter (bigger model, longer training — slower but better):
  python train.py --dataset shakespeare --layers 4 --emb 128 --context 128 --steps 8000

Let it run for a fixed amount of time instead of guessing a step count —
useful for "leave it training overnight" runs:
  python train.py --dataset everything --layers 8 --emb 384 --heads 8 --context 192 --hours 24

Then talk to it:
  python generate.py --model checkpoints/shakespeare.npz --prompt "ROMEO:"

Press ctrl-c any time — progress is saved before exiting.
"""

import argparse
import os
import time

# A model this small trains FASTER on one thread: numpy's default
# multi-threading spends more time coordinating than computing on matrices
# this size. (Set OMP_NUM_THREADS yourself before running to override.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from download_data import get_dataset
from model import GPT, Adam, Config, save_checkpoint, load_checkpoint
from tokenizer import CharTokenizer

rng = np.random.default_rng(0)
BASE = os.path.dirname(os.path.abspath(__file__))


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


def lr_at(step, progress_frac, lr_max, warmup=200):
    """Warm up, then cosine-decay — starts gentle, ends gentle. Standard.
    progress_frac is 0 at the start of training and 1 at the planned end,
    measured in whichever unit training is bounded by (steps, or time)."""
    if step < warmup:
        return lr_max * (step + 1) / warmup
    return 0.1 * lr_max + 0.45 * lr_max * (1 + np.cos(np.pi * min(progress_frac, 1.0)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="shakespeare or code (auto-downloads)")
    p.add_argument("--data", help="path to any .txt file to learn from")
    p.add_argument("--out", help="checkpoint path (default checkpoints/<name>)")
    p.add_argument("--resume", help="checkpoint to continue training from")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--hours", type=float,
                   help="train for this many hours instead of a fixed step "
                        "count (overrides --steps; still checkpoints/samples "
                        "every 500 steps and every ~10 minutes)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--emb", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-3)
    args = p.parse_args()

    # ---- data ----
    if args.data:
        if not os.path.exists(args.data):
            raise SystemExit(f"can't find '{args.data}' — check the path (and "
                             f"run commands from inside the project folder)")
        path, name = args.data, os.path.splitext(os.path.basename(args.data))[0]
    elif args.dataset:
        path, name = get_dataset(args.dataset), args.dataset
    else:
        raise SystemExit("pick data:  --dataset shakespeare | --dataset code | --data file.txt")
    text = open(path, encoding="utf-8", errors="ignore").read()

    # ---- model (fresh, or resumed from a saved brain) ----
    if args.resume:
        if not os.path.exists(args.resume):
            raise SystemExit(f"can't find '{args.resume}' — check the path")
        model, chars = load_checkpoint(args.resume)
        tok = CharTokenizer(chars)
        print(f"resumed {args.resume}  (size flags are ignored when resuming "
              f"— the saved brain keeps its shape)")
        missing = sorted(set(text) - set(chars))
        if missing:
            shown = "".join(missing[:20]) + ("…" if len(missing) > 20 else "")
            print(f"warning: {len(missing)} character(s) in this text are not "
                  f"in the saved brain's alphabet and will be SKIPPED: {shown!r}\n"
                  f"         (a brain's alphabet is fixed the first time it is "
                  f"trained — for very different text, train a fresh brain)")
    else:
        if args.emb % args.heads:
            raise SystemExit(f"--emb must be divisible by --heads "
                             f"({args.emb} doesn't split into {args.heads} equal heads)")
        tok = CharTokenizer.from_text(text)
        cfg = Config(vocab_size=tok.vocab_size, context=args.context,
                     n_layer=args.layers, n_head=args.heads, n_emb=args.emb)
        model = GPT(cfg)

    ids = tok.encode(text)
    split = int(0.95 * len(ids))
    train_ids, valid_ids = ids[:split], ids[split:]
    ctx = model.cfg.context

    # each training example needs context+1 characters; insist on headroom so
    # both the training and validation splits can always fill a batch
    if len(valid_ids) < ctx + 2 or len(train_ids) < 20 * (ctx + 2):
        raise SystemExit(
            f"'{path}' has only {len(ids):,} usable characters — too short to "
            f"train with --context {ctx}. Use a file with at least "
            f"~{25 * (ctx + 2):,} characters, or lower --context.")

    out = args.out or os.path.join(BASE, "checkpoints", name)
    if not out.endswith(".npz"):
        out += ".npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"data: {len(ids):,} characters, vocab {tok.vocab_size} | "
          f"model: {model.num_params():,} parameters | "
          f"random-guess loss would be {np.log(tok.vocab_size):.2f}")

    # ---- the loop: forward -> loss -> backward -> Adam nudge ----
    if args.hours:
        total_seconds = args.hours * 3600
        print(f"training for {args.hours:g} hours (ctrl-c stops early; "
              f"it saves every 500 steps and every ~10 minutes either way)")
    opt = Adam()
    t0 = time.time()
    last_save = t0
    step = 0
    try:
        while True:
            elapsed = time.time() - t0
            if args.hours:
                if elapsed >= total_seconds:
                    break
                progress = elapsed / total_seconds
            else:
                if step > args.steps:
                    break
                progress = step / max(1, args.steps)

            x, y = get_batch(train_ids, args.batch, ctx)
            _, loss = model.forward(x, y)
            model.backward()
            opt.step(model.params_and_grads(), lr_at(step, progress, args.lr))

            if step % 100 == 0:
                speed = (step + 1) / max(elapsed, 1e-9)
                target = f"{args.hours:g}h" if args.hours else args.steps
                remaining = (f"{(total_seconds - elapsed) / 60:.0f} min left"
                            if args.hours else f"step {step}/{args.steps}")
                print(f"step {step:5d} (target {target})   train loss {loss:.3f}   "
                      f"({speed:.1f} steps/s, {remaining})")
            due_by_steps = step and step % 500 == 0
            due_by_time = time.time() - last_save >= 600   # every ~10 min
            if due_by_steps or (args.hours and due_by_time):
                last_save = time.time()
                vl = val_loss(model, valid_ids, args.batch, ctx)
                print(f"          validation loss {vl:.3f}  (loss on text it has "
                      f"never seen — the honest score)")
                save_checkpoint(out, model, tok.chars)
                seed = list(tok.encode(text[:ctx])) or [0]
                sample = model.generate(seed, 150)
                print(f"          sample: {tok.decode(sample)!r}\n")
            step += 1
    except KeyboardInterrupt:
        print("\nstopped early — saving what it has learned so far")

    save_checkpoint(out, model, tok.chars)
    hint = next((ln for ln in text[:300].splitlines() if ln.strip()), "the")
    hint = hint[:20].replace('"', "'")
    print(f"\nbrain saved to {out}")
    print(f'try it:  python generate.py --model "{out}" --prompt "{hint}"')


if __name__ == "__main__":
    main()
