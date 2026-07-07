"""Local dashboard for pretrain-lab: live run status, loss curve, roadmap, shared notes.

Stdlib only. Parses the training logs, queries nvidia-smi, and serves a single
page at http://127.0.0.1:7871 that auto-refreshes. Notes are stored in
dashboard_notes.json so human and Claude sessions can leave each other messages.

  python dashboard.py --port 7871
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("dashboard")

ROOT = Path(__file__).resolve().parent
NOTES_PATH = ROOT / "dashboard_notes.json"

RUNS = [
    {
        "name": "fw-124m",
        "label": "GPT-2 124M on FineWeb-Edu",
        "log": "logs/fw124m_train.log",
        "target_iters": 4800,
        "tokens_per_iter": 16 * 1024 * 32,
    },
]

RE_LOSS = re.compile(r"iter (\d+): loss ([\d.]+), (\d+) tok/s")
RE_EVAL = re.compile(r"iter (\d+): train loss ([\d.]+), val loss ([\d.]+)")
RE_DONE = re.compile(r"done .*best val loss ([\d.]+)")
RE_PREP = re.compile(r"(\d+)M tokens written")


def parse_train_log(path: Path) -> dict:
    out: dict = {"losses": [], "evals": [], "done": False, "best_val": None, "tps": 0}
    if not path.exists():
        return out
    tps_samples: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := RE_LOSS.search(line):
            out["losses"].append([int(m[1]), float(m[2])])
            if int(m[3]) > 0:
                tps_samples.append(int(m[3]))
        elif m := RE_EVAL.search(line):
            out["evals"].append([int(m[1]), float(m[2]), float(m[3])])
        elif m := RE_DONE.search(line):
            out["done"] = True
            out["best_val"] = float(m[1])
    if tps_samples:
        recent = sorted(tps_samples[-5:])
        out["tps"] = recent[len(recent) // 2]
    if len(out["losses"]) > 600:  # thin for the chart, always keep the tail
        stride = len(out["losses"]) // 500
        out["losses"] = out["losses"][::stride] + out["losses"][-3:]
    return out


def run_status(run: dict) -> dict:
    parsed = parse_train_log(ROOT / run["log"])
    last_iter = parsed["losses"][-1][0] if parsed["losses"] else 0
    eta_min = None
    if not parsed["done"] and parsed["tps"] > 0:
        eta_min = (run["target_iters"] - last_iter) * run["tokens_per_iter"] / parsed["tps"] / 60
    ckpt = ROOT / "runs" / run["name"] / "ckpt.pt"
    val = parsed["evals"][-1][2] if parsed["evals"] else None
    return {
        **run,
        **parsed,
        "last_iter": last_iter,
        "eta_min": round(eta_min) if eta_min else None,
        "latest_val": parsed["best_val"] or val,
        "ckpt_exists": ckpt.exists(),
    }


def prep_status() -> dict:
    path = ROOT / "logs" / "fineweb_prep.log"
    if not path.exists():
        return {"state": "missing", "tokens_m": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = RE_PREP.findall(text)
    tokens_m = int(matches[-1]) if matches else 0
    done = "done:" in text
    return {"state": "done" if done else "running", "tokens_m": 2000 if done else tokens_m}


def gpu_status() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        util, used, total, temp = (int(v.strip()) for v in out.stdout.strip().split(","))
        return {"util": util, "vram_used": used, "vram_total": total, "temp": temp}
    except Exception:
        return None


def build_roadmap(main_run: dict) -> list[dict]:
    sft_done = (ROOT / "runs" / "fw-124m-sft" / "ckpt.pt").exists()
    hs_path = ROOT / "runs" / "fw-124m" / "hellaswag.json"
    hs = json.loads(hs_path.read_text(encoding="utf-8")) if hs_path.exists() else None
    pretrain_status = "done" if main_run["done"] else ("running" if main_run["losses"] else "queued")
    return [
        {"label": "Stage 1 — pretrain 124M on FineWeb-Edu (2B tokens)", "status": pretrain_status,
         "detail": f"iter {main_run['last_iter']}/{main_run['target_iters']}"
                   + (f", val loss {main_run['latest_val']:.3f}" if main_run["latest_val"] else "")},
        {"label": "Modern architecture (RoPE + RMSNorm + SwiGLU)", "status": "done",
         "detail": "train.py --arch modern — validated, use for the next run"},
        {"label": "Stage 2 — instruction-tune into a chat model (sft.py)",
         "status": "done" if sft_done else ("ready" if main_run["done"] else "queued"),
         "detail": "python sft.py --ckpt-dir runs/fw-124m --out-dir runs/fw-124m-sft"},
        {"label": "HellaSwag benchmark (eval_hellaswag.py)",
         "status": "done" if hs else ("ready" if main_run["done"] else "queued"),
         "detail": f"acc_norm {hs['acc_norm']:.1%} on {hs['n']} examples (random 25%)" if hs
                   else "python eval_hellaswag.py --ckpt-dir runs/fw-124m"},
        {"label": "Stage 3 — RL on verifiable tasks (grpo.py, synthetic verifier)",
         "status": "done" if (ROOT / "runs" / "fw-124m-rl" / "ckpt.pt").exists()
                   else ("ready" if sft_done else "queued"),
         "detail": "python grpo.py --ckpt-dir runs/fw-124m-sft --out-dir runs/fw-124m-rl"},
    ]


def read_notes() -> list[dict]:
    if NOTES_PATH.exists():
        return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    return []


def append_note(author: str, text: str) -> None:
    notes = read_notes()
    notes.append({"ts": time.strftime("%Y-%m-%d %H:%M"), "author": author[:40], "text": text[:2000]})
    NOTES_PATH.write_text(json.dumps(notes, indent=1), encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = (ROOT / "index.html").read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            runs = [run_status(r) for r in RUNS]
            payload = {
                "prep": prep_status(),
                "runs": runs,
                "gpu": gpu_status(),
                "roadmap": build_roadmap(runs[0]),
                "notes": read_notes(),
                "now": time.strftime("%H:%M:%S"),
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/notes":
            self._send(404, b"not found", "text/plain")
            return
        length = min(int(self.headers.get("Content-Length", 0)), 10_000)
        try:
            data = json.loads(self.rfile.read(length))
            append_note(str(data.get("author", "anon")), str(data["text"]))
            self._send(200, b'{"ok":true}', "application/json")
        except (json.JSONDecodeError, KeyError):
            self._send(400, b'{"ok":false}', "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7871)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log.info("pretrain-lab dashboard on http://127.0.0.1:%d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
