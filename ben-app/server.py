"""Ben's drawing kit: a tiny web app that collects photos of Ben's drawings, one per puppet part.

Python standard library only (runs on talmonlab under /ben, prefix stripped by Caddy).
Uploads and downloads need the family key (?k=...); only its SHA-256 is stored here.

  python server.py 5301          # env DATA = folder for the photos
API (all need ?k=):
  GET  api/state                   -> {"singer": {"head": {"t": ..., "bytes": ...}, ...}}
  POST api/upload/<char>/<part>    body = the image (JPEG)
  GET  files/<char>/<part>.jpg     -> the latest photo of that part
"""
import hashlib
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

KEY_HASH = "4d4cdd3e1725431df90be9159c5ee3ae5ce1cef576685b866cee6fc9ae8976e1"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "data"))
NAME = re.compile(r"^[a-z0-9-]{1,40}$")
MAX_BYTES = 25 * 1024 * 1024


def ok_key(q):
    k = (q.get("k") or [""])[0]
    return hashlib.sha256(k.encode()).hexdigest() == KEY_HASH


class H(BaseHTTPRequestHandler):
    def send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        parts = [p for p in u.path.split("/") if p]
        if not parts or parts == ["index.html"]:
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        if not ok_key(q):
            return self.send(403, b"wrong or missing key")
        if parts == ["api", "state"]:
            state = {}
            if os.path.isdir(DATA):
                for ch in sorted(os.listdir(DATA)):
                    d = os.path.join(DATA, ch)
                    if not (os.path.isdir(d) and NAME.match(ch)):
                        continue
                    state[ch] = {}
                    for fn in os.listdir(d):
                        if fn.endswith(".jpg") and NAME.match(fn[:-4]):
                            st = os.stat(os.path.join(d, fn))
                            state[ch][fn[:-4]] = {"t": int(st.st_mtime), "bytes": st.st_size}
            return self.send(200, json.dumps(state).encode(), "application/json")
        if len(parts) == 3 and parts[0] == "files" and parts[2].endswith(".jpg"):
            ch, part = parts[1], parts[2][:-4]
            p = os.path.join(DATA, ch, part + ".jpg")
            if NAME.match(ch) and NAME.match(part) and os.path.isfile(p):
                with open(p, "rb") as f:
                    return self.send(200, f.read(), "image/jpeg")
        return self.send(404, b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        parts = [p for p in u.path.split("/") if p]
        if not ok_key(q):
            return self.send(403, b"wrong or missing key")
        if len(parts) != 4 or parts[:2] != ["api", "upload"] or not (NAME.match(parts[2]) and NAME.match(parts[3])):
            return self.send(404, b"not found")
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BYTES:
            return self.send(413, b"bad size")
        body = self.rfile.read(n)
        if body[:2] != b"\xff\xd8":
            return self.send(415, b"send a JPEG")
        d = os.path.join(DATA, parts[2])
        os.makedirs(os.path.join(d, "history"), exist_ok=True)
        with open(os.path.join(d, parts[3] + ".jpg"), "wb") as f:
            f.write(body)
        with open(os.path.join(d, "history", f"{parts[3]}-{int(time.time())}.jpg"), "wb") as f:
            f.write(body)
        return self.send(200, b"ok")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "5301"))
    os.makedirs(DATA, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
