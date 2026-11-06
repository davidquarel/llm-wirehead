"""Live dashboard for the weight-space-programming runs.

Every POLL_S seconds it runs probe.py on each machine in runs.json (over ssh, the probe sent on stdin, so nothing has
to be installed there) and serves the merged snapshot at /api/snapshot plus the page at /.

  python3 dash/server.py [--port 8771]        (binds 127.0.0.1; expose it with a tunnel)
runs.json is re-read on every poll, so runs can be added or removed without a restart."""
import argparse, json, os, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
POLL_S = 6
SNAP = {"time": None, "hosts": {}}
LOCK = threading.Lock()


def probe(host, dirs):
    code = open(os.path.join(HERE, "probe.py")).read()
    cmd = ["python3", "-", *dirs] if host == "local" else \
          ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/wspdash-%C", "-o", "ControlPersist=900",   # one connection per host
           host, "python3 - " + " ".join(dirs)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=60)
        line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("{")), None)   # skip login banners
        if line is None:
            return {"error": (p.stderr or p.stdout)[-400:] or "no output", "latency": round(time.time() - t0, 1)}
        d = json.loads(line); d["latency"] = round(time.time() - t0, 1)
        return d
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency": round(time.time() - t0, 1)}


def poll_forever():
    while True:
        try:
            cfg = json.load(open(os.path.join(HERE, "runs.json")))
            with ThreadPoolExecutor(len(cfg)) as ex:
                res = dict(zip(cfg, ex.map(lambda h: probe(h, cfg[h]["runs"]), cfg)))
            for h, d in res.items():
                d["label"] = cfg[h].get("label", h); d["configured"] = cfg[h]["runs"]
            with LOCK:
                SNAP["hosts"] = {**{h: v for h, v in SNAP["hosts"].items() if h in cfg},
                                 **{h: (d if "error" not in d or h not in SNAP["hosts"]
                                        else {**SNAP["hosts"][h], "error": d["error"], "configured": d["configured"]}) for h, d in res.items()}}
                SNAP["time"] = time.time()
        except Exception as e:
            print("poll failed:", e, flush=True)
        time.sleep(POLL_S)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        import gzip
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        if "gzip" in (self.headers.get("Accept-Encoding") or ""):
            b = gzip.compress(b, 5); self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/").rsplit("/", 1)[-1]    # works behind a path prefix too
        if path == "snapshot":
            with LOCK:
                self._send(json.dumps(SNAP), "application/json")
        elif path in ("", "index.html") or not path.startswith("api"):
            self._send(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8771); a = ap.parse_args()
    threading.Thread(target=poll_forever, daemon=True).start()
    print(f"dashboard on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
