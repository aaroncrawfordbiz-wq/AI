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

from ask_claude import ask, looks_like_needs_real_ai
from calculator import calculate, looks_like_math
from free_search import answer_via_browser as free_search_answer
from model import load_checkpoint
from tokenizer import CharTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))


def load_banned_words(path):
    """Read the boundary list: one word per line, # lines are comments."""
    if not path or not os.path.exists(path):
        return []
    words = []
    for line in open(path, encoding="utf-8"):
        w = line.strip().lower()
        if w and not w.startswith("#"):
            words.append(w)
    return words


def make_guard(tok, banned):
    """Build the boundary: vetoes any character that would complete a banned
    word. Checked BEFORE each character is shown, so a banned word can never
    appear — the model is forced to choose different words instead."""
    def guard(ids, nxt):
        tail = tok.decode(ids[-40:] + [nxt]).lower()
        for w in banned:
            if tail.endswith(w):
                start = len(tail) - len(w)
                # only block a real word start ("class" is fine even if
                # "ass" is banned), but blocking word ENDINGS must be
                # strict, since the rest of the word hasn't been written yet
                if start == 0 or not tail[start - 1].isalpha():
                    return False
        return True
    return guard


def run(model, tok, prompt, tokens, temperature, top_k, guard=None):
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
                   guard=guard,
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
    p.add_argument("--banned", default=os.path.join(BASE, "banned_words.txt"),
                   help="boundary file of words it may never say "
                        "(default: banned_words.txt; use --banned '' to disable)")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"can't find '{args.model}' — train a brain first "
                         f"(python train.py --dataset shakespeare) or check the path")
    model, chars = load_checkpoint(args.model)
    tok = CharTokenizer(chars)
    print(f"loaded {args.model}: {model.num_params():,} parameters, "
          f"vocab {tok.vocab_size}")

    banned = load_banned_words(args.banned)
    guard = make_guard(tok, banned) if banned else None
    if banned:
        print(f"boundary active: {len(banned)} banned word(s) it can never say "
              f"(edit {args.banned} to change)")
    print()

    if args.interactive:
        print("type a prompt and the model continues it (ctrl-c to quit)")
        print("real arithmetic like '47 * 83' is answered by an actual "
              "calculator, not the model — see calculator.py")
        print("questions needing CURRENT info (prices, news, 'right now') "
              "are sent to a real AI with live web search — see ask_claude.py\n")
        while True:
            try:
                prompt = input("you> ")
            except (KeyboardInterrupt, EOFError):
                print()
                break
            expr = looks_like_math(prompt)
            if expr is not None:
                try:
                    print(f"(calculator, not the model): {expr} = {calculate(expr)}")
                except (ValueError, ZeroDivisionError, SyntaxError) as e:
                    print(f"(calculator couldn't parse that: {e})")
                print()
                continue
            if looks_like_needs_real_ai(prompt):
                has_key = bool(os.environ.get("ANTHROPIC_API_KEY")
                              or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
                if has_key:
                    print("(asking a real AI, not the trained model — this "
                          "needs current information a saved checkpoint can't have)")
                    print(ask(prompt))
                else:
                    print("(no ANTHROPIC_API_KEY set — falling back to a free, "
                          "no-key web search instead. Less capable than a real "
                          "AI, but genuinely free — see free_search.py)")
                    print(free_search_answer(prompt))
                print()
                continue
            run(model, tok, prompt, args.tokens, args.temperature, args.top_k,
                guard)
            print()
    else:
        run(model, tok, args.prompt, args.tokens, args.temperature, args.top_k,
            guard)


if __name__ == "__main__":
    main()
