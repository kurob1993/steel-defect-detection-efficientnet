#!/usr/bin/env python3
"""Serve ./data with wide-open CORS for Label Studio image loading.

Run:
  python serve_label_studio_images.py
Then import predictions_http.json into a Label Studio instance opened via http://localhost:8080.
"""
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "data"
    handler = partial(CORSRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("0.0.0.0", 8000), handler)
    print(f"Serving {root} at http://localhost:8000 with CORS enabled")
    server.serve_forever()
