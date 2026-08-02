"""
Turns a plain-English instruction like "add a red cube at 2 0 1 size 1.5"
into a structured, exact action a 3D program can execute.

Important honesty note, read this before wiring it to Blender/Unreal: this
parser uses ordinary keyword matching, NOT the trained neural network. That
is deliberate. model.py cannot reliably produce correct geometry, coordinates,
or code — it only learned what code-shaped TEXT looks like, not what it
means (see README's "Honest limits"). Handing 3D creation to it would mean
wrong shapes in wrong places with no way to know until you looked. A real,
exact parser is what actually works, and it's what real "AI does 3D content"
products use for anything that has to be geometrically correct — the
language model decides intent, ordinary code executes it exactly.

Grammar (case-insensitive, order-flexible):
  add a <shape> [<color>] [at X Y Z] [size N]

  <shape>  cube | sphere | cylinder | cone | plane
  <color>  red, green, blue, yellow, orange, purple, white, black, gray
  X Y Z    three numbers (default 0 0 0)
  N        one number, uniform scale (default 1)

Multiple commands: one per line, or separated by ';'.
"""

import re

SHAPES = {"cube", "sphere", "cylinder", "cone", "plane"}
COLORS = {
    "red": (0.8, 0.05, 0.05), "green": (0.05, 0.7, 0.05), "blue": (0.05, 0.1, 0.8),
    "yellow": (0.9, 0.85, 0.05), "orange": (0.9, 0.45, 0.05), "purple": (0.5, 0.05, 0.7),
    "white": (0.9, 0.9, 0.9), "black": (0.03, 0.03, 0.03), "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
}

_NUM = r"-?\d+(?:\.\d+)?"


class CommandError(ValueError):
    pass


def parse_command(text):
    """Parse ONE instruction. Returns a dict:
    {shape, color(name), rgb(tuple), location(x,y,z), scale}."""
    t = text.strip().lower()
    if not t:
        raise CommandError("empty command")

    shape = next((s for s in SHAPES if re.search(rf"\b{s}\b", t)), None)
    if shape is None:
        raise CommandError(f"no recognized shape in {text!r} — "
                           f"expected one of: {', '.join(sorted(SHAPES))}")

    color_name = next((c for c in COLORS if re.search(rf"\b{c}\b", t)), None)

    loc = (0.0, 0.0, 0.0)
    m = re.search(rf"\bat\s+({_NUM})\s+({_NUM})\s+({_NUM})", t)
    if m:
        loc = tuple(float(x) for x in m.groups())

    scale = 1.0
    m = re.search(rf"\b(?:size|scale)\s+({_NUM})", t)
    if m:
        scale = float(m.group(1))
        if scale <= 0:
            raise CommandError(f"size must be positive, got {scale}")

    return {
        "shape": shape,
        "color": color_name,
        "rgb": COLORS.get(color_name, (0.7, 0.7, 0.7)),
        "location": loc,
        "scale": scale,
    }


def parse_commands(text):
    """Parse many instructions, one per line or ';'-separated. Blank lines
    and '#' comments are skipped. Returns (commands, errors) — errors are
    (line_number, original_text, message) so a typo in line 5 doesn't
    throw away lines 1-4."""
    lines = [ln for raw in text.splitlines() for ln in raw.split(";")]
    commands, errors = [], []
    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            commands.append(parse_command(line))
        except CommandError as e:
            errors.append((i, line, str(e)))
    return commands, errors


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or input("command> ")
    cmds, errs = parse_commands(text)
    for c in cmds:
        print(c)
    for i, line, msg in errs:
        print(f"line {i} ({line!r}): {msg}", file=sys.stderr)
