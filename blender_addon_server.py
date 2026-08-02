"""
Runs INSIDE Blender. Opens a small local command server so an outside AI
(via blender_mcp_server.py) can ask Blender to build real objects.

Why this needs its own file, separate from blender_bridge.py: bpy calls are
only safe to make on Blender's MAIN thread. A network server has to listen
on a BACKGROUND thread (so it doesn't freeze Blender's UI while waiting for
connections). This file bridges the two the standard, safe way real Blender
addons do it:

  background thread  ->  puts each incoming command on a queue
  Blender's main thread  ->  a timer (bpy.app.timers) drains the queue and
                              is the ONLY place that ever calls bpy

HOW TO RUN (needs Blender; not testable here since this environment doesn't
have Blender installed):
  1. Open Blender -> Scripting tab.
  2. Open this file (or paste it into a new text block).
  3. Press "Run Script". You should see "Blender command server listening
     on 127.0.0.1:8765" printed in Blender's system console
     (Window > Toggle System Console on Windows; it's just your terminal
     on Mac/Linux, wherever you launched Blender from).
  4. Leave Blender open. Now run blender_mcp_server.py separately (see its
     own docstring) and connect an AI to it.

TEST WITHOUT BLENDER: this file's HTTP layer works standalone too (it falls
back to blender_bridge.py's dry-run mode when bpy isn't available), which
is how this repo's automated checks verify it without owning Blender:
    python blender_addon_server.py --dry-run
  then, in another terminal:
    curl -X POST http://127.0.0.1:8765/command -d '{"text":"red cube at 0 0 0"}'
"""

import json
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from blender_bridge import HAVE_BPY, build_command
from content_commands import parse_commands

HOST, PORT = "127.0.0.1", 8765
_inbox = queue.Queue()     # background thread -> main thread
_outbox = {}                # request id -> result, filled in on the main thread
_next_id = 0
_lock = threading.Lock()


def _submit(text):
    """Called from the server thread: queue work for the main thread and
    wait for it to be done. Blocking here is fine — it's a background
    thread, not Blender's UI thread."""
    global _next_id
    with _lock:
        req_id = _next_id
        _next_id += 1
    done = threading.Event()
    _inbox.put((req_id, text, done))
    done.wait(timeout=30)
    return _outbox.pop(req_id, {"error": "timed out waiting for Blender's main thread"})


def _drain_queue():
    """Runs ON Blender's main thread (registered as a bpy.app.timers
    callback). This is the ONLY function in this file allowed to call bpy."""
    while not _inbox.empty():
        req_id, text, done = _inbox.get()
        commands, errors = parse_commands(text)
        built = []
        for cmd in commands:
            if HAVE_BPY:
                build_command(cmd)
            built.append(cmd)
        _outbox[req_id] = {
            "built": built,
            "errors": [{"line": i, "text": t, "message": m} for i, t, m in errors],
        }
        done.set()
    return 0.1  # bpy.app.timers: reschedule in 0.1s


def _drain_queue_standalone():
    """Same as _drain_queue but for --dry-run / no-Blender testing, where
    there's no bpy.app.timers main-thread loop to piggyback on."""
    while True:
        req_id, text, done = _inbox.get()   # blocks until work arrives
        commands, errors = parse_commands(text)
        for cmd in commands:
            print(f"[dry-run] would create {cmd['shape']} at {cmd['location']}, "
                  f"scale {cmd['scale']}")
        _outbox[req_id] = {
            "built": commands,
            "errors": [{"line": i, "text": t, "message": m} for i, t, m in errors],
        }
        done.set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep Blender's console quiet

    def do_POST(self):
        if self.path != "/command":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            result = _submit(body.get("text", ""))
            payload = json.dumps(result).encode()
            self.send_response(200)
        except Exception as e:
            payload = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        payload = json.dumps({"status": "ok", "blender": HAVE_BPY}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


def start_server():
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"Blender command server listening on {HOST}:{PORT}")
    return httpd


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    start_server()
    if HAVE_BPY and not dry_run:
        import bpy
        bpy.app.timers.register(_drain_queue)
        print("registered main-thread drain timer (bpy present)")
    else:
        threading.Thread(target=_drain_queue_standalone, daemon=True).start()
        print("no bpy — running in standalone/dry-run mode "
              "(commands print instead of building real Blender objects)")
        try:
            threading.Event().wait()   # keep the process alive
        except KeyboardInterrupt:
            pass
