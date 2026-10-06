"""Static server for the landing page that makes browsers re-check every file.

`python -m http.server` sends no Cache-Control header, so browsers keep CSS and JS for hours
(guessed from the file's age) and edits do not show on reload. `no-cache` still lets the
server answer 304 Not Modified; it only forces the check.

    python super-travel/serve.py [port]        (default 5173)
"""
from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def make_server(port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("", port), partial(Handler, directory=str(Path(__file__).parent)))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
    print(f"Serving super-travel on http://localhost:{port}")
    make_server(port).serve_forever()
