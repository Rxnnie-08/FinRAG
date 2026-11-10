"""PDF ingestion (needed for FinanceBench, whose source filings are PDFs).

PDFs have real page numbers, so citations from this path carry exact pages. Table geometry is NOT
recovered here (text extraction flattens tables); use the HTML path for structure-aware table work.
"""
from __future__ import annotations

import re

from finrag.core.schema import Chunk, FilingMeta

from .chunker import split_sentences

_ITEM_LINE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)[\.\:\s\-–—]+([A-Z][^\n]{0,100})$", re.I)


def chunk_pdf(meta: FilingMeta, max_chars: int = 1800) -> list[Chunk]:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PDF ingestion needs pypdf: pip install pypdf") from e
    reader = PdfReader(meta.path)
    chunks: list[Chunk] = []
    item, section = "", ""
    seq = 0
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            m = _ITEM_LINE.match(ln)
            if m and len(ln) < 120:
                item, section = f"Item {m.group(1).upper()}", m.group(2).strip().rstrip(".")
        page_text = "\n".join(lines)
        if len(page_text) < 25:
            continue
        for piece in split_sentences(page_text, max_chars) if "\n" not in page_text else _split_lines(page_text, max_chars):
            seq += 1
            chunks.append(Chunk(
                chunk_id=f"{meta.doc_id}#p{page_no:03d}-{seq:04d}", text=piece, company=meta.company, ticker=meta.ticker,
                filing_type=meta.filing_type, fiscal_year=meta.fiscal_year, fiscal_period=meta.fiscal_period,
                period_end=meta.period_end, accession=meta.accession, source=meta.path, item=item, section=section,
                element_type="text", page=page_no, block_start=page_no, block_end=page_no, url=meta.url))
    return chunks


def _split_lines(text: str, max_chars: int) -> list[str]:
    out, cur = [], ""
    for ln in text.splitlines():
        if cur and len(cur) + 1 + len(ln) > max_chars:
            out.append(cur)
            cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        out.append(cur)
    return out
