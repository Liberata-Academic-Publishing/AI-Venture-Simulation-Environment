"""Local, passive live dashboard for the venture simulation.

Run it and open the page in a browser:

    python dashboard_server.py
    # then visit http://127.0.0.1:8000

A stdlib-only HTTP server (no third-party deps) serves a single static page and
a Server-Sent Events stream at ``/events``. Each ``/events`` connection builds a
fresh, deterministic simulation, runs it with a small per-step delay (so a human
can actually watch a sub-second run), and streams one JSON payload per simulated
day. Everything stays on localhost; nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from run_simulation import NUM_DAYS, build_simulation
from streaming import StreamingHistory

HOST = "127.0.0.1"
PORT = 8000
STEP_DELAY = 0.1  # seconds between simulated days, so the run is watchable

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        # A client closing an SSE connection (refresh / navigate away) makes the
        # keep-alive read or a mid-stream write raise these. They're expected, so
        # end the thread quietly instead of dumping a traceback to the console.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif self.path.startswith("/static/"):
            self._serve_static(os.path.basename(self.path))
        elif self.path == "/events":
            self._stream_events()
        else:
            self.send_error(404, "Not found")

    # ---- static files ----------------------------------------------------
    def _serve_static(self, name: str):
        path = os.path.join(STATIC_DIR, name)
        if not os.path.isfile(path):
            self.send_error(404, f"Missing static asset: {name}")
            return
        with open(path, "rb") as fh:
            body = fh.read()
        ext = os.path.splitext(name)[1]
        self.send_response(200)
        self.send_header("Content-Type", _MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- live SSE stream -------------------------------------------------
    def _stream_events(self):
        # The stream has no Content-Length and is effectively infinite, so don't
        # let the handler attempt a second keep-alive request on this socket
        # afterwards (that follow-up read is what raised the noisy reset).
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        history = StreamingHistory()
        env = build_simulation(history)

        try:
            self._send_event(
                {
                    "type": "meta",
                    "agents": list(history_agent_labels(env)),
                    "num_days": NUM_DAYS,
                }
            )
            for _ in range(NUM_DAYS):
                env.agentact()
                env.nextstep()
                while not history.queue.empty():
                    self._send_event(history.queue.get_nowait())
                time.sleep(STEP_DELAY)
            self._send_event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            # Browser navigated away / closed the tab; stop quietly.
            return

    def _send_event(self, payload: dict):
        data = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def log_message(self, *args):  # quieter console
        return


def history_agent_labels(env) -> list[str]:
    """Display labels for the env's agents, matching History's labelling."""
    return [getattr(a, "name", f"Agent {i}") for i, a in enumerate(env.agents, start=1)]


def main():
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Live dashboard at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
