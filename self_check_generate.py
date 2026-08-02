"""
Generate code, actually check it with a real syntax checker (never the
model itself — it can't judge its own correctness, see code_checker.py),
and if it fails, resample and try again — a real generate/check/retry loop.

Read this before expecting too much: this checks SYNTAX only ("is this
valid, parseable code"), not whether the code does what you asked for.
At this model's size, more attempts make it more likely to stumble into
something syntactically valid — they do NOT make it understand your
request better. A syntax-valid result can still be completely wrong code.
This is the honest ceiling: real verification + retries closes the
"is it even valid" gap, not the "does it do the right thing" gap.

Usage:
  python self_check_generate.py --model checkpoints/code.npz --prompt "def add(a, b):" --lang python
  python self_check_generate.py --model checkpoints/code.npz --prompt "def " --lang python --max-attempts 20
"""

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from code_checker import check_syntax
from model import load_checkpoint
from tokenizer import CharTokenizer


def generate_and_check(model, tok, prompt, lang, tokens, max_attempts,
                       base_temperature, top_k, verbose=True):
    """Repeatedly generate from `prompt` and run it through a REAL checker.
    Returns (code, ok, message, attempts_used). ok is None if this
    language has no available checker (nothing to verify against)."""
    ids = list(tok.encode(prompt)) or [0]
    best = None
    for attempt in range(1, max_attempts + 1):
        # nudge temperature up slightly each retry so it doesn't just
        # regenerate the same failing text over and over
        temperature = base_temperature + 0.05 * (attempt - 1)
        out_ids = model.generate(ids, tokens, temperature=temperature, top_k=top_k)
        code = prompt + tok.decode(out_ids)

        ok, message = check_syntax(code, lang)
        if verbose:
            status = "no checker available" if ok is None else ("PASS" if ok else "FAIL")
            print(f"attempt {attempt}/{max_attempts} (temp {temperature:.2f}): {status} — {message}")

        if ok is None:
            return code, None, message, attempt   # can't verify at all — stop, say so
        if ok:
            return code, True, message, attempt
        best = (code, ok, message)

    return (*best, max_attempts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--lang", default="python",
                   help="python, javascript, c, or cpp (real checkers for these)")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--max-attempts", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"can't find '{args.model}'")
    model, chars = load_checkpoint(args.model)
    tok = CharTokenizer(chars)
    print(f"loaded {args.model}: {model.num_params():,} parameters\n")

    code, ok, message, attempts = generate_and_check(
        model, tok, args.prompt, args.lang, args.tokens,
        args.max_attempts, args.temperature, args.top_k)

    print(f"\n--- result after {attempts} attempt(s) ---")
    if ok is None:
        print(f"couldn't verify: {message}")
    elif ok:
        print(f"PASSED a real syntax check ({message}). "
              f"Remember: syntax-valid, not proven to do what you asked.")
    else:
        print(f"never passed in {attempts} attempts — last real error: {message}")
    print("\ngenerated code:\n" + "-" * 40)
    print(code)
    print("-" * 40)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
