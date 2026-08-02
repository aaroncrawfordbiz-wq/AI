"""
Connects this project to Unreal Engine: turns plain-English shape commands
into real actors placed in an Unreal level.

Same honesty note as blender_bridge.py: content_commands.py's exact keyword
parser decides what to build, never the trained neural network — model.py
cannot reliably produce correct code or coordinates. See that file's
docstring and the README's "Honest limits" section.

HOW TO RUN THIS FOR REAL (needs Unreal Editor with Python Editor Script
Plugin enabled; not testable here since this environment doesn't have
Unreal installed):
  1. Edit > Plugins > enable "Python Editor Script Plugin" (restart if asked).
  2. Window > Developer Tools > Output Log, then the "Cmd" dropdown -> Python.
  3. In the Python console: exec(open(r"C:/path/to/unreal_bridge.py").read())
     (edit COMMANDS at the bottom of this file first, or import and call
     build_scene("red cube at 0 0 0") directly from the console).

TEST WITHOUT UNREAL (what this repo's automated tests use):
    python unreal_bridge.py --dry-run "red cube at 0 0 0; blue sphere size 200"
  Note Unreal's default unit is centimeters, unlike Blender's meters — a
  "size 200" cube is roughly a 2-meter cube.
"""

import sys

from content_commands import parse_commands

try:
    import unreal
    HAVE_UNREAL = True
except ImportError:
    HAVE_UNREAL = False

# Unreal ships these basic shape meshes with every project.
ENGINE_MESH = {
    "cube": "/Engine/BasicShapes/Cube.Cube",
    "sphere": "/Engine/BasicShapes/Sphere.Sphere",
    "cylinder": "/Engine/BasicShapes/Cylinder.Cylinder",
    "cone": "/Engine/BasicShapes/Cone.Cone",
    "plane": "/Engine/BasicShapes/Plane.Plane",
}


def build_command(cmd):
    """Spawn ONE real static-mesh actor in the current Unreal level. Only
    called when the unreal module is actually available (i.e. running
    inside the Unreal Editor's Python console)."""
    mesh = unreal.EditorAssetLibrary.load_asset(ENGINE_MESH[cmd["shape"]])
    location = unreal.Vector(*cmd["location"])
    actor = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, location)
    actor.set_actor_scale3d(unreal.Vector(cmd["scale"], cmd["scale"], cmd["scale"]))
    if cmd["color"]:
        actor.set_actor_label(f"{cmd['color']}_{cmd['shape']}")
        # A real material tint needs a dynamic material instance built from
        # a project-specific parent material, which varies per project —
        # left as a documented next step rather than guessed here.
    return actor


def build_scene(text, dry_run=False):
    """Parse and build every command in `text`. Returns (built, errors)."""
    commands, errors = parse_commands(text)
    for i, line, msg in errors:
        print(f"skipped line {i} ({line!r}): {msg}", file=sys.stderr)

    built = []
    for cmd in commands:
        if dry_run or not HAVE_UNREAL:
            print(f"[dry-run] would spawn {cmd['shape']}"
                  f"{' (' + cmd['color'] + ')' if cmd['color'] else ''} "
                  f"at {cmd['location']}, scale {cmd['scale']}")
        else:
            build_command(cmd)
            print(f"spawned {cmd['shape']} at {cmd['location']}")
        built.append(cmd)
    return built, errors


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]

    COMMANDS = " ".join(argv) if argv else """
        red cube at -200 0 0 size 100
        blue sphere at 0 0 0 size 120
        green cylinder at 200 0 0 size 100
    """
    build_scene(COMMANDS, dry_run=dry_run or not HAVE_UNREAL)
