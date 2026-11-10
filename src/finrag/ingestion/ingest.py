"""Glue: manifest <-> filings <-> chunks."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from finrag.core.fiscal import fiscal_period_for_date, parse_fye, KNOWN_FYE
from finrag.core.schema import Chunk, FilingMeta

from .chunker import ChunkerConfig, chunk_blocks
from .htmlparse import parse_html
from .pdfload import chunk_pdf
from .xbrl import xbrl_chunks


def complete_meta(meta: FilingMeta) -> FilingMeta:
    """Fill fiscal_year / fiscal_period from period_end and the company's fiscal calendar if missing."""
    if meta.fiscal_year is not None and meta.fiscal_period:
        return meta
    if meta.filing_type == "8-K" or not meta.period_end:
        return meta
    fye = meta.fiscal_year_end or KNOWN_FYE.get(meta.ticker.upper(), "1231")
    fye_month, _ = parse_fye(fye)
    fy, q = fiscal_period_for_date(date.fromisoformat(meta.period_end), fye_month)
    if meta.fiscal_year is None:
        meta.fiscal_year = fy
    if not meta.fiscal_period:
        meta.fiscal_period = "FY" if meta.filing_type == "10-K" else f"Q{q}"
    return meta


def ingest_filing(meta: FilingMeta, cfg: ChunkerConfig | None = None) -> list[Chunk]:
    meta = complete_meta(meta)
    path = Path(meta.path)
    if not path.exists():
        raise FileNotFoundError(f"Filing not found: {path}")
    if path.suffix.lower() == ".pdf":
        return chunk_pdf(meta)
    html = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_blocks(parse_html(html), meta, cfg)
    chunks += xbrl_chunks(html, meta)
    return chunks


def ingest_manifest(entries: list[FilingMeta], cfg: ChunkerConfig | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for m in entries:
        out.extend(ingest_filing(m, cfg))
    return out


def load_manifest(path: str | Path, base_dir: str | Path | None = None) -> list[FilingMeta]:
    path = Path(path)
    base = Path(base_dir) if base_dir else path.parent
    entries = [FilingMeta.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    for e in entries:
        if e.path and not Path(e.path).is_absolute():
            e.path = str((base / e.path).resolve())
    return [complete_meta(e) for e in entries]


def save_manifest(entries: list[FilingMeta], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([e.to_dict() for e in entries], indent=2), encoding="utf-8")
