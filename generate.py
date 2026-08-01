"""
Talk to a trained brain. Loads a checkpoint and generates text from your
prompt, streaming character by character — the model literally writing live.

  python generate.py --model checkpoints/shakespeare.npz --prompt "ROMEO:"
  python generate.py --model checkpoints/code.npz --prompt "def " --tokens 400
  python generate.py --model checkpoints/shakespeare.npz --interactive

Knobs:
  --temperature  0.4 = safe and repetitive, 1.0 = creative and chaotic
  --top_k        only sample among the k most likely next characters
"""

import argparse
import os
import sys

# same single-thread speedup as train.py — must happen before numpy loads
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from model import load_checkpoint
from tokenizer import CharTokenizer


def run(model, tok, prompt, tokens, temperature, top_k):
    known = sum(c in tok.ctoi for c in prompt)
    if known < len(prompt):
        print(f"(note: {len(prompt) - known} character(s) of your prompt "
              f"aren't in this model's alphabet and were ignored)")
    ids = list(tok.encode(prompt))
    if not ids:
        ids = list(tok.encode(" ")) or [0]
    sys.stdout.write(prompt)
    sys.stdout.flush()
    model.generate(ids, tokens, temperature=temperature, top_k=top_k,
                   stream=lambda i: (sys.stdout.write(tok.decode([i])),
                                     sys.stdout.flush()))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint .npz from train.py")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--interactive", action="store_true",
                   help="keep prompting it in a loop")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"can't find '{args.model}' — train a brain first "
                         f"(python train.py --dataset shakespeare) or check the path")
    model, chars = load_checkpoint(args.model)
    tok = CharTokenizer(chars)
    print(f"loaded {args.model}: {model.num_params():,} parameters, "
          f"vocab {tok.vocab_size}\n")

    if args.interactive:
        print("type a prompt and the model continues it (ctrl-c to quit)\n")
        while True:
            try:
                prompt = input("you> ")
            except (KeyboardInterrupt, EOFError):
                print()
                break
            run(model, tok, prompt, args.tokens, args.temperature, args.top_k)
            print()
    else:
        run(model, tok, args.prompt, args.tokens, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
