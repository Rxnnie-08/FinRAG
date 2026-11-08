"""Command line: `finrag <command>`. Run `finrag --help`."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from finrag.core.config import Settings
from finrag.core.schema import Answer


def _settings(a) -> Settings:
    s = Settings.from_env()
    if getattr(a, "home", None):
        s.home = Path(a.home)
    for k in ("embedder", "llm"):
        if getattr(a, k, None):
            setattr(s, k, getattr(a, k))
    return s


def print_answer(ans: Answer, verbose: bool = False) -> None:
    print(f"\nQ: {ans.question}")
    if ans.rewrite and ans.rewrite.mode != "off":
        print(f"   read as: {ans.rewrite.standalone}")
        if ans.rewrite.filters:
            print(f"   filters: {ans.rewrite.filters}")
        for a in ans.rewrite.assumptions:
            print(f"   assumption: {a}")
    print(f"\nA: {ans.text}\n")
    for c in ans.citations:
        where = " · ".join(x for x in (c.item, c.note or c.section, c.table_id, f"p.{c.page}" if c.page else "") if x)
        print(f"  [{c.n}] {c.label}   ({where})")
        if c.span and verbose:
            print(f"       “{c.span[:220]}”")
    if ans.support_rate is not None:
        print(f"\n  claims supported: {ans.support_rate:.0%} | cited: {ans.citation_coverage:.0%} | {ans.mode} | {ans.generator} | {ans.timings_ms.get('total', 0):.0f} ms")
    for w in ans.warnings:
        print(f"  ! {w}")


def cmd_ingest(a) -> None:
    from finrag.ingestion import load_manifest
    from finrag.pipeline import RagPipeline
    s = _settings(a)
    filings = load_manifest(a.manifest)
    pipe = RagPipeline.build(filings, s)
    info = pipe.info()
    print(f"Indexed {info['n_chunks']} chunks from {info['n_filings']} filings -> {s.index_dir} (embedder: {info['embedder']})")
    gold = Path(a.manifest).parent / "qa_gold.jsonl"
    if gold.exists():
        shutil.copy(gold, s.home / "qa_gold.jsonl")


def cmd_demo(a) -> None:
    root = Path(__file__).resolve().parents[2]
    sample = root / "sample_data"
    if not (sample / "manifest.json").exists():
        subprocess.run([sys.executable, str(sample / "generate.py")], check=True)
    a.manifest = str(sample / "manifest.json")
    cmd_ingest(a)
    print("Demo corpus is SYNTHETIC (fictional companies). Try:  finrag ask \"What were Northwind Robotics' total assets in fiscal 2024?\"  or  finrag serve")


def cmd_fetch(a) -> None:
    from finrag.ingestion import save_manifest
    from finrag.ingestion.edgar import EdgarClient
    s = _settings(a)
    client = EdgarClient(s.sec_user_agent)
    out = Path(a.out)
    metas = []
    for t in a.tickers:
        for m in client.list_filings(t, tuple(a.forms), a.limit):
            print(f"downloading {m.ticker} {m.filing_type} {m.period_end} ...")
            metas.append(client.download(m, out / "filings"))
    save_manifest(metas, out / "manifest.json")
    print(f"Wrote {out / 'manifest.json'}. Next: finrag ingest --manifest {out / 'manifest.json'}")


def cmd_ask(a) -> None:
    from finrag.pipeline import RagPipeline
    pipe = RagPipeline.load(_settings(a))
    print_answer(pipe.ask(a.question, mode=a.mode, rewrite=a.rewrite, k=a.k), verbose=a.verbose)


def cmd_serve(a) -> None:
    from finrag.pipeline import RagPipeline
    from finrag.server.app import create_app
    s = _settings(a)
    app = create_app(RagPipeline.load(s), s)
    print(f"FinRAG UI on http://{a.host}:{a.port}  (Ctrl+C to stop)")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


def cmd_eval(a) -> None:
    from finrag.evaluation import load_gold, run_benchmark, to_markdown
    from finrag.evaluation.judge import HeuristicJudge, LLMJudge
    from finrag.evaluation.runner import CONFIGS, save_results
    from finrag.generation import make_llm
    from finrag.pipeline import RagPipeline
    s = _settings(a)
    pipe = RagPipeline.load(s)
    gold = load_gold(a.gold or (s.home / "qa_gold.jsonl"))
    if a.limit:
        gold = gold[: a.limit]
    judge = LLMJudge(make_llm(a.judge)) if a.judge else HeuristicJudge()
    cfgs = [c for c in CONFIGS if not a.configs or c["name"] in a.configs]
    res = run_benchmark(pipe, gold, cfgs, k=a.k, judge=judge, progress=lambda n, i, t: print(f"\r{n}: {i}/{t}", end="", flush=True) if i == t or i % 10 == 0 else None)
    print()
    paths = save_results(res, a.out or (s.home / "eval"))
    print(to_markdown(res))
    print("saved:", ", ".join(str(p) for p in paths.values()))


def cmd_financebench(a) -> None:
    from finrag.evaluation.financebench import prepare
    m, g = prepare(a.dest, a.limit_docs)
    print(f"Wrote {m} and {g}. Next:\n  finrag ingest --manifest {m} --home {a.dest}/home\n  finrag eval --home {a.dest}/home --gold {g}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="finrag", description="RAG question answering over SEC filings with source attribution")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, embedder=False, llm=False):
        sp.add_argument("--home", help="data folder (default: $FINRAG_HOME or ./finrag_data)")
        if embedder:
            sp.add_argument("--embedder", help="lsa | st:<model> | ollama:<model>")
        if llm:
            sp.add_argument("--llm", help="none | anthropic:<model> | openai:<model>@<base_url>")

    sp = sub.add_parser("demo", help="generate the synthetic sample corpus and index it"); common(sp, True); sp.set_defaults(fn=cmd_demo)
    sp = sub.add_parser("ingest", help="parse + chunk + index the filings listed in a manifest"); common(sp, True)
    sp.add_argument("--manifest", required=True); sp.set_defaults(fn=cmd_ingest)
    sp = sub.add_parser("fetch", help="download filings from SEC EDGAR (needs SEC_USER_AGENT)"); common(sp)
    sp.add_argument("--tickers", nargs="+", required=True); sp.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    sp.add_argument("--limit", type=int, default=2, help="filings per company"); sp.add_argument("--out", default="edgar_data"); sp.set_defaults(fn=cmd_fetch)
    sp = sub.add_parser("ask", help="ask one question"); common(sp, False, True)
    sp.add_argument("question"); sp.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"])
    sp.add_argument("--rewrite", default="rules", choices=["off", "rules", "llm"]); sp.add_argument("-k", type=int, default=6)
    sp.add_argument("-v", "--verbose", action="store_true"); sp.set_defaults(fn=cmd_ask)
    sp = sub.add_parser("serve", help="start the web UI"); common(sp, False, True)
    sp.add_argument("--host", default="127.0.0.1"); sp.add_argument("--port", type=int, default=8000); sp.set_defaults(fn=cmd_serve)
    sp = sub.add_parser("eval", help="run the retrieval/answer benchmark"); common(sp, False, True)
    sp.add_argument("--gold"); sp.add_argument("--out"); sp.add_argument("-k", type=int, default=5); sp.add_argument("--limit", type=int)
    sp.add_argument("--configs", nargs="+"); sp.add_argument("--judge", help="LLM spec for LLM-as-judge, e.g. anthropic:claude-sonnet-5-5"); sp.set_defaults(fn=cmd_eval)
    sp = sub.add_parser("financebench", help="download FinanceBench questions + PDFs and convert to this project's format")
    sp.add_argument("--dest", default="financebench_data"); sp.add_argument("--limit-docs", type=int); sp.set_defaults(fn=cmd_financebench)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
