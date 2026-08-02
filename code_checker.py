"""
A REAL checker: actually verifies code with the real tool for that
language (Python's own compiler, node's parser, gcc's parser — never the
trained model judging its own work, since it has no way to actually know).

This only checks SYNTAX — "is this valid, parseable code" — not whether
the code does what you asked. That second bar (does it actually walk, jump,
throw a stone correctly) needs real understanding of intent, which no
amount of retrying gives a model that never had it. Syntax-valid is the
honest ceiling for automatic self-checking at this project's scale.
"""

import ast
import shutil
import subprocess
import tempfile

# Each checker returns (ok: bool, message: str). Anything not listed here
# has no available checker (returns None so callers can say so honestly,
# instead of silently skipping verification).
def _check_python(code):
    try:
        ast.parse(code)
        return True, "valid Python syntax"
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"


def _check_with_external(code, suffix, cmd_fn, tool_name):
    if shutil.which(cmd_fn[0]) is None:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(cmd_fn + [path], capture_output=True,
                                text=True, timeout=10)
        if result.returncode == 0:
            return True, f"valid syntax ({tool_name})"
        return False, (result.stderr or result.stdout).strip()[:300]
    except subprocess.TimeoutExpired:
        return False, f"{tool_name} timed out"


CHECKERS = {
    "python": lambda code: _check_python(code),
    "javascript": lambda code: _check_with_external(
        code, ".js", ["node", "--check"], "node"),
    "c": lambda code: _check_with_external(
        code, ".c", ["gcc", "-fsyntax-only"], "gcc"),
    "cpp": lambda code: _check_with_external(
        code, ".cpp", ["g++", "-fsyntax-only"], "g++"),
}


def check_syntax(code, lang):
    """Returns (ok, message). ok is None if no checker is available for
    this language on this machine — that's an honest 'can't verify', not
    a pass or a fail."""
    checker = CHECKERS.get(lang)
    if checker is None:
        return None, f"no checker registered for '{lang}'"
    result = checker(code)
    if result is None:
        return None, f"'{lang}' checker tool isn't installed on this machine"
    return result
