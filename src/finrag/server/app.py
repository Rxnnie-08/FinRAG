"""Flask app: JSON API + the single-page UI in ./static.

Binds to localhost by default and has NO authentication: do not expose it to a network as is.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from finrag.core.config import Settings
from finrag.pipeline import MODES, RagPipeline

STATIC = Path(__file__).parent / "static"
EXAMPLES = [
    "What were Northwind Robotics' total assets at the end of fiscal 2024?",
    "How much debt principal does Northwind have maturing in 2027?",
    "What was Harbor Lane Foods' revenue in Q2 of fiscal 2025?",
    "What material weakness did Harbor Lane Foods identify in fiscal 2024?",
    "What was NWRX capex in fiscal 2024?",
    "What was Apple's total assets in fiscal 2024?",
]


def create_app(pipe: RagPipeline, settings: Settings | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    settings = settings or Settings.from_env()
    lock = threading.Lock()
    chunk_index = {c.chunk_id: c for c in pipe.retriever.chunks}

    def err(msg: str, code: int = 400):
        return jsonify({"error": msg}), code

    def parse_ask(data: dict):
        q = (data.get("question") or "").strip()
        if not q:
            return None, "Enter a question."
        if len(q) > 1000:
            return None, "Question is too long (max 1000 characters)."
        mode = data.get("mode", "hybrid")
        rewrite = data.get("rewrite", "rules")
        if mode not in MODES:
            return None, f"mode must be one of {list(MODES)}"
        if rewrite not in ("off", "rules", "llm"):
            return None, "rewrite must be off, rules or llm"
        try:
            k = int(data.get("k", pipe.top_k))
        except (TypeError, ValueError):
            return None, "k must be an integer"
        if not 1 <= k <= 15:
            return None, "k must be between 1 and 15"
        hist = data.get("history") or []
        if not isinstance(hist, list) or not all(isinstance(h, str) for h in hist):
            return None, "history must be a list of strings"
        return (q, mode, rewrite, k, [h[:1000] for h in hist[-3:]]), None

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC, name)

    @app.get("/api/status")
    def status():
        info = pipe.info()
        info["examples"] = EXAMPLES
        info["has_gold"] = (settings.home / "qa_gold.jsonl").exists()
        info["synthetic"] = all(f.accession.startswith("SYN-") for f in pipe.filings) if pipe.filings else False
        return jsonify(info)

    @app.post("/api/ask")
    def ask():
        parsed, e = parse_ask(request.get_json(silent=True) or {})
        if e:
            return err(e)
        q, mode, rewrite, k, hist = parsed
        if rewrite == "llm" and pipe.llm is None:
            return err("rewrite=llm needs an LLM backend (set FINRAG_LLM).")
        return jsonify(pipe.ask(q, mode=mode, rewrite=rewrite, k=k, history=hist).to_dict())

    @app.post("/api/compare")
    def compare():
        parsed, e = parse_ask(request.get_json(silent=True) or {})
        if e:
            return err(e)
        q, _, rewrite, k, hist = parsed
        return jsonify(pipe.compare(q, k=k, rewrite=rewrite, history=hist))

    @app.get("/api/chunk/<path:chunk_id>")
    def chunk(chunk_id: str):
        c = chunk_index.get(chunk_id)
        return jsonify({**c.to_dict(), "label": c.label}) if c else err("unknown chunk", 404)

    @app.get("/api/eval/latest")
    def eval_latest():
        p = settings.home / "eval" / "results.json"
        if not p.exists():
            return jsonify({"available": False})
        res = json.loads(p.read_text())
        res.pop("rows", None)
        res["available"] = True
        return jsonify(res)

    @app.post("/api/eval/run")
    def eval_run():
        from finrag.evaluation import load_gold, run_benchmark
        from finrag.evaluation.runner import save_results
        gp = settings.home / "qa_gold.jsonl"
        if not gp.exists():
            return err("No gold set found at " + str(gp), 404)
        if not lock.acquire(blocking=False):
            return err("A benchmark is already running.", 409)
        try:
            res = run_benchmark(pipe, load_gold(gp), k=5)
            save_results(res, settings.home / "eval")
        finally:
            lock.release()
        res.pop("rows", None)
        res["available"] = True
        return jsonify(res)

    return app
