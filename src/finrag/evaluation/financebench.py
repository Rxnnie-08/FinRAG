"""FinanceBench (Islam et al., 2023) loader: 150 open-source questions over 84 SEC filings.

`prepare()` downloads the question file and the source PDFs from the public patronus-ai/financebench repo and writes
a manifest + gold file in this project's format. Notes that matter for your report:
  * FinanceBench ships PDFs; PDF text extraction has real page numbers but flattens tables. For table-geometry
    experiments, map questions to the EDGAR HTML filing instead (the doc names encode company/year/form).
  * evidence_page_num is 0-indexed in FinanceBench; chunk.page is 1-indexed, so gold pages are shifted by +1.
  * The question file's `company` field supplies the real name ("American Express", not AMERICANEXPRESS) so the query
    rewriter can resolve it; FB_FYE holds non-December fiscal year-ends. `fye_overrides` adds or corrects entries.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

from finrag.core.fiscal import month_end, parse_fye, quarter_end
from finrag.core.schema import FilingMeta
from finrag.ingestion.ingest import save_manifest

RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
# Fiscal year-end month/day by FinanceBench doc-name prefix (month level; 52/53-week years rounded to month end).
# Values are from memory of each company's reporting calendar: check any you plan to quote in the report.
FB_FYE = {"ADOBE": "1130", "AMCOR": "0630", "BESTBUY": "0131", "COSTCO": "0831", "FOOTLOCKER": "0131", "GENERALMILLS": "0531",
          "MICROSOFT": "0630", "NIKE": "0531", "ULTA": "0131", "WALMART": "0131"}
_DOC = re.compile(r"^(?P<co>.+?)_(?P<yr>\d{4})(?P<q>Q[1-4])?_(?P<form>10K|10Q|8K|EARNINGS)", re.I)


def parse_doc_name(doc_name: str, fye_overrides: dict[str, str] | None = None, company: str = "") -> FilingMeta | None:
    m = _DOC.match(doc_name)
    if not m:
        return None
    co, yr, q, form = m.group("co"), int(m.group("yr")), (m.group("q") or "").upper(), m.group("form").upper()
    ftype = {"10K": "10-K", "10Q": "10-Q", "8K": "8-K", "EARNINGS": "8-K"}[form]   # earnings releases/calls are filed as 8-K exhibits
    fye = {**FB_FYE, **(fye_overrides or {})}.get(co.upper(), "1231")
    fm, _ = parse_fye(fye)
    if ftype == "10-Q" and q:
        end = quarter_end(fm, yr, int(q[1]))
    elif form == "EARNINGS" and q:
        end = quarter_end(fm, yr, int(q[1]))
    else:
        end = month_end(yr, fm)
    return FilingMeta(company=company or co.replace("&", " & "), ticker=co.upper(), filing_type=ftype, period_end=end.isoformat(),
                      fiscal_year=yr if (ftype != "8-K" or form == "EARNINGS") else None, fiscal_period="FY" if ftype == "10-K" else (q or ("Q?" if ftype == "10-Q" else "")),
                      fiscal_year_end=fye, accession=doc_name)


def convert(rows: list[dict], pdf_dir: str | Path, fye_overrides: dict[str, str] | None = None) -> tuple[list[FilingMeta], list[dict]]:
    pdf_dir = Path(pdf_dir)
    filings: dict[str, FilingMeta] = {}
    gold: list[dict] = []
    for r in rows:
        meta = parse_doc_name(r["doc_name"], fye_overrides, r.get("company", ""))
        if meta is None:
            continue
        meta.path = str(pdf_dir / f"{r['doc_name']}.pdf")
        filings.setdefault(r["doc_name"], meta)
        gold.append({
            "id": r["financebench_id"], "question": r["question"], "answer": r["answer"],
            "kind": (r.get("question_reasoning") or r["question_type"] or "").lower().replace(" ", "_")[:40],
            "gold_doc": meta.doc_id, "gold_filing": {"ticker": meta.ticker, "filing_type": meta.filing_type, "fiscal_year": meta.fiscal_year},
            "evidence_pages": sorted({e["evidence_page_num"] + 1 for e in r.get("evidence", []) if e.get("evidence_page_num") is not None}),
            "evidence": [], "justification": r.get("justification", "")})
    return list(filings.values()), gold


def prepare(dest: str | Path, limit_docs: int | None = None, fye_overrides: dict[str, str] | None = None, session=None) -> tuple[Path, Path]:
    """Download data + PDFs, write manifest.json and qa_gold.jsonl into dest. Returns their paths."""
    sess = session or requests.Session()
    dest = Path(dest)
    pdf_dir = dest / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    jl = sess.get(f"{RAW}/data/financebench_open_source.jsonl", timeout=60)
    jl.raise_for_status()
    rows = [json.loads(l) for l in jl.text.splitlines() if l.strip()]
    if limit_docs:
        keep = list(dict.fromkeys(r["doc_name"] for r in rows))[:limit_docs]
        rows = [r for r in rows if r["doc_name"] in keep]
    for name in dict.fromkeys(r["doc_name"] for r in rows):
        target = pdf_dir / f"{name}.pdf"
        if not target.exists():
            resp = sess.get(f"{RAW}/pdfs/{name}.pdf", timeout=120)
            resp.raise_for_status()
            target.write_bytes(resp.content)
    filings, gold = convert(rows, pdf_dir, fye_overrides)
    save_manifest(filings, dest / "manifest.json")
    gp = dest / "qa_gold.jsonl"
    gp.write_text("\n".join(json.dumps(g) for g in gold) + "\n", encoding="utf-8")
    return dest / "manifest.json", gp
