"""Flask app: JSON API + the single-page UI in ./static.

Binds to localhost by default and has NO authentication: do not expose it to a network as is.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

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
    app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
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

    def generic_examples() -> list[str]:
        real = [f for f in pipe.filings if not f.accession.startswith("SYN-")] or pipe.filings
        if not real:
            return []
        f = real[-1]
        who = f.company
        when = f"fiscal {f.fiscal_year}" if f.fiscal_year else f"the period ended {f.period_end}"
        return [f"What were {who}'s total assets in {when}?", f"What was {who}'s net income in {when}?",
                f"How much cash and cash equivalents did {who} have in {when}?", f"What was {who}'s revenue in {when}?",
                f"What was {who}'s capex in {when}?"]

    @app.get("/api/status")
    def status():
        info = pipe.info()
        info["synthetic"] = all(f.accession.startswith("SYN-") for f in pipe.filings) if pipe.filings else False
        info["examples"] = EXAMPLES if info["synthetic"] else generic_examples()
        # the shipped gold set describes the synthetic corpus only; it says nothing about uploaded filings
        info["has_gold"] = info["synthetic"] and (settings.home / "qa_gold.jsonl").exists()
        return jsonify(info)

    @app.errorhandler(413)
    def too_big(_):
        return err("That file is larger than 80 MB.", 413)

    @app.post("/api/upload")
    def upload():
        from finrag.ingestion import detect_meta
        f = request.files.get("file")
        if not f or not f.filename:
            return err("Choose a filing to add.")
        ext = Path(f.filename).suffix.lower()
        if ext not in (".htm", ".html", ".pdf"):
            return err("Only .htm, .html or .pdf filings are supported.")
        dest = settings.filings_dir / "uploads"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{uuid4().hex[:8]}_{secure_filename(f.filename) or 'filing' + ext}"
        f.save(path)
        try:
            meta, notes = detect_meta(path, {k: request.form.get(k, "") for k in ("company", "ticker", "filing_type", "period_end", "fiscal_year_end")},
                                      original_name=f.filename)
            if not lock.acquire(blocking=False):
                raise RuntimeError("Another upload or benchmark is running. Try again in a moment.")
            try:
                result = pipe.add_filing(meta)
            finally:
                lock.release()
        except FileExistsError as e:
            path.unlink(missing_ok=True)
            return err(str(e), 409)
        except (ValueError, RuntimeError) as e:
            path.unlink(missing_ok=True)
            return err(str(e), 422)
        chunk_index.update({c.chunk_id: c for c in pipe.retriever.chunks})
        result["notes"] = notes
        return jsonify(result)

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
        res = json.loads(p.read_text(encoding="utf-8"))
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
