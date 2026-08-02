"""
The MCP server: this is the piece a real AI (Claude Desktop, Claude Code, or
any other MCP-capable client) connects to. It exposes Blender as a small set
of TOOLS the AI can call by name — it does NOT use this repo's trained
model.py at all. That's the point: a big, real language model is genuinely
good at picking tool calls from a plain-English request; the tiny model we
built from scratch earlier in this project is not, which is why 3D creation
was never routed through it (see content_commands.py's docstring).

Architecture (three separate processes talking to each other):

  You, typing in Claude       Claude reads this file's tool          Blender,
  Desktop / Claude Code   ---> descriptions, calls e.g.        --->  actually
  in plain English             build_shapes(commands=...)            building
                                over MCP (this file)                  the object
                                      |
                                      v
                          blender_addon_server.py's HTTP API
                          (running inside Blender, see its docstring)

SETUP:
  1. pip install "mcp>=1.6,<2"
  2. Open Blender, run blender_addon_server.py inside it (see that file's
     docstring) and leave Blender open.
  3. Add this server to your AI client's MCP config. For Claude Desktop,
     edit claude_desktop_config.json:
       {
         "mcpServers": {
           "blender": {
             "command": "python",
             "args": ["/full/path/to/blender_mcp_server.py"]
           }
         }
       }
     Restart the client. It will now show three tools: build_shapes,
     list_shape_options, get_blender_status.
  4. Just ask, in plain English: "build a small red house out of cubes"
     — the AI breaks that into shape commands and calls build_shapes.

TEST WITHOUT AN AI CLIENT OR BLENDER (what this repo's automated checks
use — runs the tool functions directly, no MCP protocol, no bpy needed):
    python blender_mcp_server.py --selftest
"""

import sys
import urllib.error
import urllib.request

from content_commands import COLORS, SHAPES

ADDON_URL = "http://127.0.0.1:8765/command"
STATUS_URL = "http://127.0.0.1:8765/"


def _call_blender(text, timeout=10):
    """Forward a batch of shape commands to blender_addon_server.py running
    inside Blender. Returns its JSON result, or a clear error if Blender's
    server isn't reachable (e.g. Blender isn't open, or the addon script
    wasn't run) instead of a confusing connection traceback."""
    import json
    try:
        req = urllib.request.Request(
            ADDON_URL, data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        return {"error": f"can't reach Blender's command server at {ADDON_URL} "
                         f"({e}) — is Blender open with blender_addon_server.py "
                         f"running inside it?"}


def _get_status(timeout=5):
    import json
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        return {"reachable": False, "error": str(e)}


def build_shapes_impl(commands: str) -> dict:
    """Implementation shared by the MCP tool and --selftest, so both are
    exercising the exact same code path."""
    return _call_blender(commands)


try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("blender-bridge")

    @mcp.tool()
    def build_shapes(commands: str) -> dict:
        """Build real objects in the currently open Blender scene.

        `commands` is one instruction per line (or ';'-separated), e.g.:
          "red cube at -2 0 0 size 1
           blue sphere at 0 0 0 size 1.2
           green cylinder at 2 0 0"

        Each instruction is: add a <shape> [<color>] [at X Y Z] [size N].
        Call list_shape_options first if unsure what's available.
        Returns which shapes were actually built and any lines that failed
        to parse (with the reason), so you can see and fix mistakes.
        """
        return build_shapes_impl(commands)

    @mcp.tool()
    def list_shape_options() -> dict:
        """List the exact shapes and color names build_shapes understands."""
        return {"shapes": sorted(SHAPES), "colors": sorted(COLORS)}

    @mcp.tool()
    def get_blender_status() -> dict:
        """Check whether Blender's command server is reachable right now."""
        return _get_status()

except ImportError:
    mcp = None   # allows --selftest to still run without the mcp package


def selftest():
    print("shapes:", sorted(SHAPES))
    print("colors:", sorted(COLORS))
    status = _get_status()
    print("blender status:", status)
    result = build_shapes_impl("red cube at 0 0 0; not a shape")
    print("build_shapes result:", result)
    assert "error" in result or "built" in result, "unexpected response shape"
    print("PASS — MCP tool functions run end to end")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif mcp is None:
        raise SystemExit('the mcp package is not installed — run: '
                         'pip install "mcp>=1.6,<2"')
    else:
        mcp.run()
