"""Benchmark runner: the ablation table for the report.

CONFIGS is the grid {dense, sparse, hybrid} x {rewrite off, rules}. Every configuration sees the same questions, the
same corpus and the same generator, so differences come from retrieval/rewriting only. Results carry bootstrap
95% confidence intervals and a paired-bootstrap comparison: with ~150 questions, differences of a few points
are noise, and the output says so.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path

from finrag.pipeline import RagPipeline

from .metrics import answer_correct, retrieval_metrics
from .judge import HeuristicJudge

CONFIGS = [
    {"name": "dense", "mode": "dense", "rewrite": "off"},
    {"name": "sparse", "mode": "sparse", "rewrite": "off"},
    {"name": "hybrid", "mode": "hybrid", "rewrite": "off"},
    {"name": "dense+rewrite", "mode": "dense", "rewrite": "rules"},
    {"name": "sparse+rewrite", "mode": "sparse", "rewrite": "rules"},
    {"name": "hybrid+rewrite", "mode": "hybrid", "rewrite": "rules"},
]
METRICS = ["recall@1", "recall@3", "recall@5", "mrr", "answer_correct", "faithful", "support_rate", "abstain_ok", "latency_ms"]


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"),) * 3
    rnd = random.Random(seed)
    means = sorted(statistics.fmean(rnd.choices(vals, k=len(vals))) for _ in range(n))
    return statistics.fmean(vals), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def paired_diff(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> dict:
    """Mean(a-b) with a paired-bootstrap 95% CI. 'significant' = the CI excludes 0."""
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    if not d:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "significant": False, "n": 0}
    m, lo, hi = bootstrap_ci(d, n, seed)
    return {"diff": m, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0, "n": len(d)}


def run_benchmark(pipe: RagPipeline, gold: list[dict], configs: list[dict] | None = None, k: int = 5, judge=None,
                  progress=None, use_llm: bool = True) -> dict:
    configs = configs or CONFIGS
    judge = judge or HeuristicJudge()
    rows: list[dict] = []
    for cfg in configs:
        for i, g in enumerate(gold):
            ans = pipe.ask(g["question"], mode=cfg["mode"], rewrite=cfg["rewrite"], k=k, use_llm=use_llm)
            chunks = [r.chunk for r in ans.retrieved]
            m = retrieval_metrics(chunks, g) if g.get("kind") != "unanswerable" else {}
            c = answer_correct(ans.text, g)
            j = judge.judge(ans, g)
            rows.append({
                "config": cfg["name"], "id": g["id"], "kind": g.get("kind", ""), "question": g["question"], **m,
                "answer_correct": None if g.get("kind") == "unanswerable" else float(c["correct"]),
                "abstain_ok": float(c["correct"]) if g.get("kind") == "unanswerable" else None,
                "answer": ans.text, "gold": g.get("answer", ""),
                "faithful": None if j.get("faithful") is None else float(j["faithful"]),
                "support_rate": ans.support_rate, "citation_coverage": ans.citation_coverage,
                "latency_ms": ans.timings_ms.get("total"),
                "retrieval_ms": ans.timings_ms.get("retrieve"),
                "filters": ans.rewrite.filters if ans.rewrite else {},
            })
            if progress:
                progress(cfg["name"], i + 1, len(gold))
    summary: dict[str, dict] = {}
    for cfg in configs:
        rs = [r for r in rows if r["config"] == cfg["name"]]
        summary[cfg["name"]] = {}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci([r.get(m) for r in rs])
            summary[cfg["name"]][m] = {"mean": mean, "lo": lo, "hi": hi, "n": sum(1 for r in rs if r.get(m) is not None)}
        kinds = sorted({r["kind"] for r in rs if r["kind"] and r["kind"] != "unanswerable"})
        summary[cfg["name"]]["by_kind"] = {kd: {
            "recall@3": statistics.fmean([r["recall@3"] for r in rs if r["kind"] == kd and r.get("recall@3") is not None] or [float("nan")]),
            "answer_correct": statistics.fmean([r["answer_correct"] for r in rs if r["kind"] == kd and r.get("answer_correct") is not None] or [float("nan")]),
            "n": sum(1 for r in rs if r["kind"] == kd)} for kd in kinds}
    by = {c["name"]: {r["id"]: r for r in rows if r["config"] == c["name"]} for c in configs}
    comparisons = {}
    for a, b in (("hybrid", "dense"), ("hybrid", "sparse"), ("hybrid+rewrite", "hybrid")):
        if a in by and b in by:
            for m in ("recall@3", "answer_correct", "mrr"):
                ids = [i for i in by[a] if by[a][i].get(m) is not None and by[b][i].get(m) is not None]
                comparisons[f"{a} − {b} [{m}]"] = paired_diff([by[a][i][m] for i in ids], [by[b][i][m] for i in ids])
    return {"n_questions": len(gold), "k": k, "judge": judge.name if hasattr(judge, "name") else "heuristic",
            "summary": summary, "comparisons": comparisons, "rows": rows}


def _fmt(s: dict, pct: bool = True) -> str:
    if s["n"] == 0 or s["mean"] != s["mean"]:
        return "n/a"
    return f"{s['mean'] * 100:.0f}% ({s['lo'] * 100:.0f}-{s['hi'] * 100:.0f})" if pct else f"{s['mean']:.0f}"


def to_markdown(res: dict) -> str:
    s = res["summary"]
    head = "| Config | Recall@1 | Recall@3 | Recall@5 | MRR | Answer correct | Faithful | Abstain ok | Latency ms |\n|---|---|---|---|---|---|---|---|---|\n"
    body = ""
    for name, m in s.items():
        body += (f"| {name} | {_fmt(m['recall@1'])} | {_fmt(m['recall@3'])} | {_fmt(m['recall@5'])} | {m['mrr']['mean']:.2f} | "
                 f"{_fmt(m['answer_correct'])} | {_fmt(m['faithful'])} | {_fmt(m['abstain_ok'])} | {_fmt(m['latency_ms'], False)} |\n")
    cmp_ = "\n**Paired comparisons (mean difference, 95% bootstrap CI):**\n\n"
    for k, v in res["comparisons"].items():
        flag = "significant" if v["significant"] else "not significant"
        cmp_ += f"- {k}: {v['diff'] * 100:+.1f} pts ({v['lo'] * 100:+.1f} to {v['hi'] * 100:+.1f}), n={v['n']}, {flag}\n"
    note = (f"\n_{res['n_questions']} questions, k={res['k']}, judge={res['judge']}. Brackets are bootstrap 95% CIs; "
            f"overlapping intervals mean the difference may be noise._\n")
    return head + body + cmp_ + note


def save_results(res: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "results.json", "md": out / "results.md", "csv": out / "rows.csv"}
    paths["json"].write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    paths["md"].write_text(to_markdown(res), encoding="utf-8")
    with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
        cols = ["config", "id", "kind", "question", "recall@1", "recall@3", "recall@5", "mrr", "answer_correct", "abstain_ok", "faithful",
                "support_rate", "citation_coverage", "latency_ms", "answer", "gold"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(res["rows"])
    return paths
