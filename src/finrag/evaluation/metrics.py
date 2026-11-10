"""Gold-set format and per-question metrics.

A gold item is a JSON object (one per line in qa_gold.jsonl):
  id, question, answer, kind (table|text|period|footnote|xbrl|unanswerable),
  gold_filing: {ticker, filing_type, fiscal_year}        which filing holds the evidence
  evidence: ["strings that must ALL appear in one relevant chunk"]          (HTML filings)
  evidence_pages + gold_doc: page numbers (1-indexed) in a PDF filing        (FinanceBench)
A retrieved chunk is RELEVANT if it comes from the gold filing and satisfies the evidence test.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from finrag.core.numbers import any_number_match, parse_numbers
from finrag.core.schema import Chunk
from finrag.generation.attribution import is_abstention
from finrag.retrieval.tokenizer import tokenize


def load_gold(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm(s: str) -> str:
    return re.sub(r"[\s ]+", " ", s.lower().replace("’", "'")).strip()


def _filing_ok(c: Chunk, g: dict) -> bool:
    gf = g.get("gold_filing") or {}
    if g.get("gold_doc"):
        return c.doc_id == g["gold_doc"]
    if gf.get("ticker") and c.ticker != gf["ticker"]:
        return False
    if gf.get("filing_type") and c.filing_type != gf["filing_type"]:
        return False
    if gf.get("fiscal_year") is not None and c.fiscal_year != gf["fiscal_year"]:
        return False
    return True


def is_relevant(c: Chunk, g: dict) -> bool:
    if g.get("kind") == "unanswerable" or not _filing_ok(c, g):
        return False
    pages = g.get("evidence_pages")
    if pages:
        return c.page in set(pages)
    ev = g.get("evidence") or []
    text = _norm(c.text + " " + " ".join(c.footnotes))
    return bool(ev) and all(_norm(e) in text for e in ev)


def retrieval_metrics(retrieved_chunks: list[Chunk], g: dict, ks=(1, 3, 5)) -> dict:
    """Context recall@k (any relevant chunk in top-k), reciprocal rank, plus filing-level hit."""
    rel = [is_relevant(c, g) for c in retrieved_chunks]
    first = next((i for i, r in enumerate(rel, start=1) if r), None)
    out = {f"recall@{k}": float(any(rel[:k])) for k in ks}
    out["mrr"] = 1.0 / first if first else 0.0
    out["filing_hit@k"] = float(any(_filing_ok(c, g) for c in retrieved_chunks))
    return out


def answer_correct(pred: str, g: dict) -> dict:
    """Correctness against the gold answer. Numeric gold answers: the number must appear (strict and scale-tolerant).
    Text gold answers: >= 60% of the gold answer's content words appear in the prediction."""
    gold = g.get("answer") or ""
    pred_body = re.sub(r"\s*\[\d{1,2}(?:\s*[,;]\s*\d{1,2})*\]", "", pred)
    if g.get("kind") == "unanswerable":
        return {"correct": is_abstention(pred_body) or "no filing matching" in pred_body.lower() or "could not find" in pred_body.lower(),
                "strict": None, "type": "abstain"}
    if is_abstention(pred_body):
        return {"correct": False, "strict": False, "type": "abstained"}
    if parse_numbers(gold):
        tol = any_number_match(pred_body, gold, scale_tolerant=True)
        strict = any_number_match(pred_body, gold, scale_tolerant=False)
        return {"correct": tol, "strict": strict, "type": "numeric"}
    gt = {t for t in tokenize(gold)}
    pt = set(tokenize(pred_body))
    recall = len(gt & pt) / (len(gt) or 1)
    return {"correct": recall >= 0.6, "strict": recall >= 0.6, "type": "text", "recall": round(recall, 3)}
