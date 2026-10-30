"""Peephole Parley dashboard: a small local web server.

    python dashboard/server.py            # then open http://127.0.0.1:8765
    python dashboard/server.py --port 9000

Only the Python standard library plus the project's own dependencies
(numpy, scikit-learn, joblib) are used. The server binds to 127.0.0.1, so it is
reachable from this computer only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")
sys.path.insert(0, ROOT)

import joblib  # noqa: E402

from cost import cost  # noqa: E402
from engine import apply_fixpoint_counted, apply_rule, explore  # noqa: E402
from features import TEXTBOOK_ORDER, extract  # noqa: E402
from generator import generate  # noqa: E402
from interpreter import equivalent  # noqa: E402
from rules import RULE_NAMES  # noqa: E402
from tac import parse, to_text  # noqa: E402
from train import predict_choice  # noqa: E402

MAX_LINES = 200
MAX_BYTES = 20_000
_lock = threading.Lock()
_bundles: dict[int, dict] = {}

STRATEGIES = [
    ("ml", "Model's pick"),
    ("fixed", "Best fixed pipeline"),
    ("oracle", "Oracle (best possible)"),
    ("textbook", "Textbook order"),
    ("fixpoint", "Repeat all rules until nothing changes"),
]


def read_json(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


def budgets_available():
    return sorted(int(os.path.basename(p)[len("portfolio_b"):-len(".json")])
                  for p in glob.glob(os.path.join(ROOT, "data", "portfolio_b*.json")))


def bundle(budget: int):
    with _lock:
        if budget not in _bundles:
            _bundles[budget] = {
                "meta": read_json("data", f"portfolio_b{budget}.json"),
                "model": joblib.load(os.path.join(ROOT, "models", f"best_model_b{budget}_full.joblib")),
            }
        return _bundles[budget]


def names(pipe):
    return [RULE_NAMES[r] for r in pipe]


def trace(code, pipe):
    """Cost after each rule of a pipeline, for the animated strip."""
    steps, cur = [], tuple(code)
    for r in pipe:
        before = cost(cur)
        cur = apply_rule(cur, r)
        steps.append({"rule": RULE_NAMES[r], "before": before, "after": cost(cur)})
    return steps, cur


def run_strategy(key, code, budget, b):
    meta, bun = b["meta"], b["model"]
    if key == "ml":
        x = extract(code)[: bun["n_features"]].reshape(1, -1)
        idx = int(predict_choice(bun["kind"], bun["model"], x, len(meta["portfolio"]))[0])
        pipe = tuple(meta["portfolio"][idx])
    elif key == "fixed":
        pipe = tuple(meta["static_pipeline"])
    elif key == "oracle":
        pipe = explore(code, budget)[1]
    elif key == "textbook":
        pipe = TEXTBOOK_ORDER[:budget]
    else:
        out, effort = apply_fixpoint_counted(code, TEXTBOOK_ORDER)
        return {"pipeline": names(TEXTBOOK_ORDER), "repeated": True, "steps": [], "effort": effort,
                "cost": cost(out), "output": to_text(out), "same_behaviour": equivalent(code, out, trials=10)}
    steps, out = trace(code, pipe)
    return {"pipeline": names(pipe), "repeated": False, "steps": steps, "effort": len(pipe),
            "cost": cost(out), "output": to_text(out), "same_behaviour": equivalent(code, out, trials=10)}


def optimize(payload):
    text = payload.get("code", "")
    budget = int(payload.get("budget", 4))
    if len(text.encode()) > MAX_BYTES or text.count("\n") > MAX_LINES:
        return 400, {"error": f"Program is too long. Keep it under {MAX_LINES} lines."}
    if budget not in budgets_available():
        return 400, {"error": f"No trained model for budget {budget}."}
    try:
        code = parse(text)
    except ValueError as e:
        return 400, {"error": f"Cannot read the program: {e}"}
    if not code or cost(code) == 0:
        return 400, {"error": "The program is empty. Paste some three-address code or pick an example."}
    b = bundle(budget)
    base = cost(code)
    return 200, {"budget": budget, "base_cost": base,
                 "results": {k: dict(run_strategy(k, code, budget, b), label=label) for k, label in STRATEGIES}}


def budget_info(budget):
    meta = read_json("data", f"portfolio_b{budget}.json")
    cmp_ = read_json("results", f"model_comparison_b{budget}_full.json")
    return {
        "budget": budget,
        "portfolio": [names(p) for p in meta["portfolio"]],
        "static": names(meta["static_pipeline"]),
        "textbook": names(meta["textbook_pipeline"]),
        "winner": cmp_["winner"],
        "cv": cmp_["cv"],
    }


def examples():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "examples", "*.tac"))):
        with open(p) as f:
            out.append({"name": os.path.splitext(os.path.basename(p))[0], "code": f.read()})
    return out


MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}


class Handler(BaseHTTPRequestHandler):
    server_version = "PeepholeParley"

    def log_message(self, fmt, *args):  # quiet
        if os.environ.get("PARLEY_VERBOSE"):
            super().log_message(fmt, *args)

    def send(self, status, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/api/summary":
                return self.send(200, {"summary": read_json("results", "summary_all.json"),
                                       "rules": list(RULE_NAMES), "budgets": budgets_available()})
            if url.path == "/api/budget":
                return self.send(200, budget_info(int(q.get("b", ["4"])[0])))
            if url.path == "/api/examples":
                return self.send(200, examples())
            if url.path == "/api/random":
                seed = int(q.get("seed", ["1"])[0])
                return self.send(200, {"seed": seed, "code": to_text(generate(seed))})
        except (ValueError, FileNotFoundError, KeyError) as e:
            return self.send(404, {"error": str(e)})
        rel = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
        path = os.path.normpath(os.path.join(STATIC, rel))
        if not path.startswith(STATIC + os.sep) or not os.path.isfile(path):
            return self.send(404, {"error": "not found"})
        with open(path, "rb") as f:
            self.send(200, f.read(), MIME.get(os.path.splitext(path)[1], "application/octet-stream"))

    def do_POST(self):
        if urlparse(self.path).path != "/api/optimize":
            return self.send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 2 * MAX_BYTES:
                return self.send(413, {"error": "Request is too large."})
            status, body = optimize(json.loads(self.rfile.read(n) or b"{}"))
            self.send(status, body)
        except (ValueError, KeyError, TypeError) as e:
            self.send(400, {"error": f"Bad request: {e}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Peephole Parley dashboard: http://127.0.0.1:{args.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
