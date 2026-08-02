"""
A real calculator — actual arithmetic, computed by Python, not the neural
network guessing digits.

Why this file exists: the trained model in model.py can only ever produce
characters that LOOK like the answer to a math problem, because it has no
concept of numbers — it just learned which digits tend to follow which
others in text it read. Ask it "what is 47 * 83?" and it will confidently
write a wrong number that merely resembles an answer.

Real AI products solve this the same way this file does: they don't make
the language model do arithmetic. They give it a TOOL — a real calculator —
and let the model's job be recognizing "this looks like a math question,"
while a normal, exact, deterministic program does the actual computing.
That hand-off (language model decides WHEN to use a tool; ordinary code
does the actual work) is the whole idea behind "AI agents" and "tool use."

Only + - * / ** ( ) and numbers are allowed — this is deliberately NOT a
general eval(), which would let arbitrary code run. See is_safe_expr().
"""

import ast
import operator

OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("not a plain arithmetic expression")


def calculate(expr):
    """Evaluate a plain arithmetic expression exactly. Raises ValueError for
    anything that isn't pure +-*/**() arithmetic on numbers — no names, no
    function calls, no attribute access, so this can never run arbitrary code."""
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def looks_like_math(text):
    """Heuristic: does this text look like an arithmetic question rather
    than a prompt for the language model? Used to decide whether to hand
    off to the real calculator instead of generating characters."""
    stripped = text.strip().rstrip("?").strip()
    lowered = stripped.lower()
    for prefix in ("what is ", "what's ", "calculate ", "compute ", "="):
        if lowered.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
            break
    if not stripped:
        return None
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in stripped):
        return None
    if not any(c.isdigit() for c in stripped):
        return None
    return stripped


if __name__ == "__main__":
    import sys
    expr = " ".join(sys.argv[1:]) or input("expression> ")
    print(calculate(expr))
