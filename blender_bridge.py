"""
Connects this project to Blender: turns plain-English shape commands into
real objects in a Blender scene.

Read content_commands.py's docstring first — the geometry is created by an
exact, ordinary parser, NOT by the trained neural network. This file is
honest about that: it never asks model.py to produce Blender code, because
model.py cannot reliably produce correct code (proven in the README).
What it CAN safely do is caption the scene — flavor text only, never
control — and that's opt-in and clearly labeled below.

HOW TO RUN THIS FOR REAL (needs Blender installed; not testable here since
this environment doesn't have Blender):
  1. Open Blender.
  2. Switch to the "Scripting" tab (top of the window).
  3. Open this file, or paste its contents into a new text block.
  4. At the BOTTOM of the file, edit COMMANDS to whatever you want built.
  5. Press "Run Script" (the play button). Objects appear in the 3D viewport.

  Or from a terminal, headless (no Blender window):
    blender --background --python blender_bridge.py -- "red cube at 0 0 0"

TEST WITHOUT BLENDER (what this repo's automated tests use):
    python blender_bridge.py --dry-run "red cube at 0 0 0; blue sphere size 2"
  Dry-run runs the exact same parsing and prints what WOULD be built,
  without needing bpy installed — useful for checking your commands before
  opening Blender.
"""

import sys

from content_commands import parse_commands

try:
    import bpy
    HAVE_BPY = True
except ImportError:
    HAVE_BPY = False

PRIMITIVE_ADD = {
    "cube": "primitive_cube_add",
    "sphere": "primitive_uv_sphere_add",
    "cylinder": "primitive_cylinder_add",
    "cone": "primitive_cone_add",
    "plane": "primitive_plane_add",
}


def build_command(cmd):
    """Create ONE real Blender object from a parsed command dict. Only
    called when bpy is actually available (i.e. running inside Blender)."""
    add_fn = getattr(bpy.ops.mesh, PRIMITIVE_ADD[cmd["shape"]])
    add_fn(location=cmd["location"])
    obj = bpy.context.active_object
    obj.scale = (cmd["scale"],) * 3

    if cmd["color"]:
        mat = bpy.data.materials.new(name=f"{cmd['color']}_material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*cmd["rgb"], 1.0)
        obj.data.materials.append(mat)
    return obj


def build_scene(text, dry_run=False):
    """Parse and build every command in `text`. Returns (built, errors)."""
    commands, errors = parse_commands(text)
    for i, line, msg in errors:
        print(f"skipped line {i} ({line!r}): {msg}", file=sys.stderr)

    built = []
    for cmd in commands:
        if dry_run or not HAVE_BPY:
            print(f"[dry-run] would create {cmd['shape']}"
                  f"{' (' + cmd['color'] + ')' if cmd['color'] else ''} "
                  f"at {cmd['location']}, scale {cmd['scale']}")
        else:
            build_command(cmd)
            print(f"created {cmd['shape']} at {cmd['location']}")
        built.append(cmd)
    return built, errors


def caption_scene(built, checkpoint):
    """OPTIONAL, purely cosmetic: ask a trained model.py brain for a short
    flavor-text description of the scene. This text is never parsed back
    into commands and never controls anything — it's just a caption."""
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from model import load_checkpoint
    from tokenizer import CharTokenizer

    model, chars = load_checkpoint(checkpoint)
    tok = CharTokenizer(chars)
    shapes = ", ".join(c["shape"] for c in built) or "an empty scene"
    seed = list(tok.encode(f"A scene with {shapes}.")) or [0]
    out = tok.decode(model.generate(seed, 120, temperature=0.8, top_k=40))
    return out


if __name__ == "__main__":
    # `blender --python this.py -- args` puts real args after a lone '--'
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]

    COMMANDS = " ".join(argv) if argv else """
        red cube at -2 0 0 size 1
        blue sphere at 0 0 0 size 1.2
        green cylinder at 2 0 0 size 1
    """
    build_scene(COMMANDS, dry_run=dry_run or not HAVE_BPY)
