"""Element-based chunking.

Rules that matter for financial filings:
  * a table is never split mid-row, and every chunk of a table repeats caption, units and column headers;
  * footnotes that belong to a table travel with it;
  * text is chunked on paragraph/sentence boundaries inside one Item/Note, never across them;
  * no content is dropped: oversized paragraphs are split on sentences, not truncated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from finrag.core.schema import Chunk, FilingMeta

from .htmlparse import Block
from .tables import ParsedTable, TableRow, table_to_markdown

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")


@dataclass
class ChunkerConfig:
    max_chars: int = 1800
    min_chars: int = 300
    table_max_chars: int = 5000
    min_keep_chars: int = 25


def split_sentences(text: str, max_chars: int) -> list[str]:
    """Split oversized text on sentence boundaries (then on words as a last resort). Nothing is dropped."""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    cur = ""
    for s in _SENT_SPLIT.split(text):
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _table_header_line(table_id: str, caption: str, note: str, units: str) -> str:
    bits = f"Table {table_id}"
    if caption:
        bits += f": {caption}"
    if note:
        bits += f" ({note})"
    if units:
        bits += f" [{units}]"
    return bits


def _table_chunks(b: Block, table_id: str, meta: FilingMeta, cfg: ChunkerConfig, next_id) -> list[Chunk]:
    pt: ParsedTable = b.table  # type: ignore[assignment]
    head = _table_header_line(table_id, b.caption, b.note, b.units)
    foot = ("Footnotes:\n" + "\n".join(b.footnotes)) if b.footnotes else ""
    rows = pt.rows

    # split rows into groups that fit; every group repeats the header
    groups: list[list[TableRow]] = []
    cur: list[TableRow] = []
    cur_len = 0
    for r in rows:
        line_len = sum(len(x) for x in [r.label] + r.cells) + 3 * (len(r.cells) + 1)
        if cur and cur_len + line_len > cfg.table_max_chars:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(r)
        cur_len += line_len
    if cur:
        groups.append(cur)

    chunks: list[Chunk] = []
    n_rows = len(rows)
    done = 0
    for gi, g in enumerate(groups):
        md = table_to_markdown(pt.header, g)
        part_note = "" if len(groups) == 1 else f"\n(rows {done + 1}-{done + len(g)} of {n_rows}; header repeated)"
        done += len(g)
        text = f"{head}{part_note}\n{md}" + (f"\n{foot}" if foot else "")
        chunks.append(Chunk(
            chunk_id=next_id(), text=text, company=meta.company, ticker=meta.ticker, filing_type=meta.filing_type,
            fiscal_year=meta.fiscal_year, fiscal_period=meta.fiscal_period, period_end=meta.period_end,
            accession=meta.accession, source=meta.path, item=b.item, section=b.section, note=b.note,
            element_type="table" if len(groups) == 1 else "table_rows", table_id=table_id, caption=b.caption,
            units=b.units, footnotes=list(b.footnotes),
            table_data={"header": pt.header, "rows": [[r.label] + r.cells for r in g], "groups": [r.group for r in g]},
            page=b.page, block_start=b.idx, block_end=b.idx, url=meta.url))
    return chunks


def chunk_blocks(blocks: list[Block], meta: FilingMeta, cfg: ChunkerConfig | None = None, start_seq: int = 0) -> list[Chunk]:
    cfg = cfg or ChunkerConfig()
    chunks: list[Chunk] = []
    seq = [start_seq]

    def next_id() -> str:
        seq[0] += 1
        return f"{meta.doc_id}#{seq[0]:05d}"

    buf: list[Block] = []
    buf_len = 0
    table_no = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        text = "\n".join(b.text for b in buf)
        first = buf[0]
        if len(text) >= cfg.min_keep_chars:
            for piece in split_sentences(text, cfg.max_chars):
                chunks.append(Chunk(
                    chunk_id=next_id(), text=piece, company=meta.company, ticker=meta.ticker,
                    filing_type=meta.filing_type, fiscal_year=meta.fiscal_year, fiscal_period=meta.fiscal_period,
                    period_end=meta.period_end, accession=meta.accession, source=meta.path, item=first.item,
                    section=first.section, note=first.note, element_type="text", page=first.page,
                    block_start=first.idx, block_end=buf[-1].idx, url=meta.url))
        buf, buf_len = [], 0

    key = None
    for b in blocks:
        if b.kind == "heading":
            flush()
            key = None
            continue
        if b.kind == "table":
            flush()
            table_no += 1
            chunks.extend(_table_chunks(b, f"T{table_no}", meta, cfg, next_id))
            continue
        k = (b.item, b.note)
        if key is not None and k != key:
            flush()
        key = k
        if buf and buf_len + len(b.text) > cfg.max_chars and buf_len >= cfg.min_chars:
            flush()
        buf.append(b)
        buf_len += len(b.text) + 1
    flush()
    return chunks
