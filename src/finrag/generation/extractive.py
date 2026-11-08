"""Offline, deterministic answerer: picks the best table row / sentence from the retrieved chunks and quotes it.

This is a lookup, not a summariser. It exists so the whole system (and its attribution checks) runs without any
API key, and as an honest baseline to compare an LLM generator against.
"""
from __future__ import annotations

import math
import re

from finrag.core.numbers import parse_numbers
from finrag.core.schema import RetrievedChunk
from finrag.retrieval.tokenizer import tokenize

_GENERIC = {"total", "million", "billion", "thousand", "fiscal", "year", "company", "amount", "much", "many", "percent",
            "percentage", "end", "ended", "report", "reported", "disclose", "disclosed", "announce", "announced", "say", "said",
            "net", "carry", "carries", "apply", "applies", "give", "show", "list", "tell"}
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(“\"])|\n+")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
ABSTAIN = "I could not find this in the retrieved filings."
_PCT_CUE = re.compile(r"\b(percent|percentage|grow\w*|increase[ds]?|decrease[ds]?|decline[ds]?|change[ds]?)\b|%", re.I)
_LEADIN = re.compile(r"(as follows|following|summarizes?|presented below|shown below)[^.]*[.:]?\s*$|:\s*$", re.I)


def has_value(text: str) -> bool:
    """True if the text carries a real quantity (not just a year or a day-of-month)."""
    for n in parse_numbers(text):
        v = abs(n.value)
        if n.pct or n.scale > 1 or "$" in n.raw or "," in n.raw or "." in n.raw:
            return True
        if float(v).is_integer() and 1900 <= v <= 2100:
            continue
        if float(v).is_integer() and v <= 31:
            continue
        return True
    return False


_QUANT = re.compile(r"\b(how much|how many|what (?:was|were|is|are)|amount|total|percent|percentage|rate|revenue|income|assets|liabilit\w+|debt|cash|eps|payments?)\b", re.I)


def entity_tokens(retrieved: list[RetrievedChunk]) -> set[str]:
    """Company-name / ticker tokens: they appear in every chunk of a filing, so they never discriminate."""
    out: set[str] = set()
    for rc in retrieved:
        out |= set(tokenize(rc.chunk.company)) | set(tokenize(rc.chunk.ticker))
    return out


def _qtokens(q: str, entity: set[str] | None = None) -> set[str]:
    return {t for t in tokenize(q) if t not in _GENERIC and t not in (entity or ())}


def _row_score(row_label: str, group: str, q: set[str]) -> float:
    toks = set(tokenize(f"{row_label} {group}"))
    toks -= {t for t in toks if t.isdigit() and len(t) <= 2}
    if not toks:
        return 0.0
    overlap = len(toks & q)
    label_overlap = len(set(tokenize(row_label)) & q)
    if label_overlap == 0:
        return 0.0
    return (overlap + label_overlap) / (2 * math.sqrt(len(set(tokenize(row_label))) or 1))


def _best_columns(header: list[str], query: str, period_end: str = "") -> list[int]:
    ql = query.lower()
    years = {m.group(0) for m in _YEAR.finditer(query)}
    cols = [i for i, h in enumerate(header[1:], start=1) if any(y in h for y in years)] if years else []
    if not cols and period_end:
        # fiscal "Q2 2025" for a June-FYE company ends in Dec 2024: use the filing's own period-end year
        cols = [i for i, h in enumerate(header[1:], start=1) if period_end[:4] in h]
    if cols:
        span = next((w for w in ("six months", "nine months", "year ended", "three months") if w in ql), "")
        if not span and any("three months" in header[i].lower() for i in cols):
            span = "three months"
        if span:
            narrowed = [i for i in cols if span in header[i].lower()]
            cols = narrowed or cols
        return cols
    qtoks = _qtokens(query)
    scored = [(len(set(tokenize(h)) & qtoks - {t for t in qtoks if t.isdigit()}), i) for i, h in enumerate(header[1:], start=1)]
    best = max((s for s, _ in scored), default=0)
    if best > 0:
        return [i for s, i in scored if s == best]
    return list(range(1, min(len(header), 3)))     # no signal: most recent two columns


def _table_candidates(rc: RetrievedChunk, query: str, q: set[str]):
    c = rc.chunk
    td = c.table_data
    if not td:
        return
    header = td["header"]
    groups = td.get("groups") or [""] * len(td["rows"])
    for row, grp in zip(td["rows"], groups):
        label, cells = row[0], row[1:]
        s = _row_score(label, grp, q)
        if s <= 0:
            continue
        cols = _best_columns(header, query, c.period_end)
        parts = []
        for ci in cols:
            if ci - 1 < len(cells) and cells[ci - 1] != "":
                parts.append(f"{header[ci]}: {cells[ci - 1]}")
        if not parts:
            continue
        unit = f" ({c.units})" if c.units else ""
        yield s, f"{label.rstrip(':')} — " + "; ".join(parts) + unit


_XLINE = re.compile(r"^(?P<label>.+?) \(us-gaap:\w+\) (?P<kind>as of|for the period) (?P<period>[\d\-]+(?: to [\d\-]+)?): (?P<value>.+)$")


def _xbrl_candidates(rc: RetrievedChunk, query: str, q: set[str]):
    years = {m.group(0) for m in _YEAR.finditer(query)} | {rc.chunk.period_end[:4]}
    rows = []
    for ln in rc.chunk.text.splitlines():
        m = _XLINE.match(ln.strip())
        if not m:
            continue
        sc = _row_score(m.group("label"), "", q)
        if sc > 0:
            rows.append((sc, m))
    if not rows:
        return
    explicit = {m.group(0) for m in _YEAR.finditer(query)}
    for sc, m in rows:
        period = m.group("period")
        year_ok = any(y in period for y in explicit) if explicit else period.endswith(rc.chunk.period_end) or rc.chunk.period_end in period
        if explicit and not year_ok:
            continue
        if not explicit and not year_ok:
            continue
        yield sc + 0.05, f"{m.group('label')} — {m.group('kind')} {period}: {m.group('value')} (XBRL)"


def _text_candidates(rc: RetrievedChunk, query: str, q: set[str]):
    qnums = {round(n.value, 4) for n in parse_numbers(query) if not _YEAR.fullmatch(str(int(abs(n.value))))}
    for sent in _SENT.split(rc.chunk.text):
        sent = sent.strip()
        if len(sent) < 20 or sent.startswith(("| ", "Table T", "Footnotes:")):
            continue
        toks = set(tokenize(sent))
        ov = len(toks & q)
        if not ov:
            continue
        s = ov / math.sqrt(max(len(q), 1) * max(len(toks), 1)) * 2.2
        if qnums & {round(n.value, 4) for n in parse_numbers(sent)}:
            s += 0.2
        if _LEADIN.search(sent):
            s *= 0.4                     # a sentence that only introduces a table is not an answer
        yield s, sent


def _footnote_candidates(rc: RetrievedChunk, query: str, q: set[str]):
    for fn in rc.chunk.footnotes:
        body = re.sub(r"^(\(\s*\w{1,2}\s*\)|\*+|†|‡|\d{1,2}\))\s*", "", fn)
        toks = set(tokenize(body))
        ov = len(toks & q)
        if ov:
            yield ov / math.sqrt(max(len(q), 1) * max(len(toks), 1)) * 2.2 + 0.05, body


def extractive_answer(query: str, retrieved: list[RetrievedChunk], max_items: int = 2, min_score: float = 0.42):
    """Return (answer_text, used_chunk_ranks) or (ABSTAIN, [])."""
    q = _qtokens(query, entity_tokens(retrieved))
    if not q or not retrieved:
        return ABSTAIN, []
    wants_number = bool(_QUANT.search(query))
    pct_cue = bool(_PCT_CUE.search(query))
    if pct_cue:
        q = q | {"increase", "decrease", "growth", "grow"}
    evidence_tokens: set[str] = set()
    for rc in retrieved:
        evidence_tokens |= set(tokenize(rc.chunk.index_text))
    absent = {t for t in q if not t.isdigit() and t not in evidence_tokens and t not in {"increase", "decrease", "growth", "grow"}}
    content = {t for t in q if not t.isdigit()}
    if len(absent) >= 2 and len(absent) / max(len(content), 1) >= 0.6:
        return ABSTAIN, []              # the question asks about terms that appear nowhere in the evidence
    cands: list[tuple[float, str, int]] = []
    for pos, rc in enumerate(retrieved):
        bonus = 0.12 / (1 + pos)
        et = rc.chunk.element_type
        gen = (_table_candidates(rc, query, q) if et in ("table", "table_rows")
               else _xbrl_candidates(rc, query, q) if et == "xbrl_facts" else _text_candidates(rc, query, q))
        if et in ("table", "table_rows"):
            gen = list(gen) + list(_footnote_candidates(rc, query, q))
        for s, text in gen:
            if wants_number:
                s += 0.2 if has_value(text) else -0.15
            if pct_cue and "%" in text:
                s += 0.3
            cands.append((s + bonus, text, pos + 1))
    cands.sort(key=lambda x: (-x[0], len(x[1])))
    if not cands or cands[0][0] < min_score:
        return ABSTAIN, []
    chosen, seen = [], set()
    for s, text, rank in cands:
        if text in seen:
            continue
        if chosen and s < cands[0][0] * 0.92:
            break
        seen.add(text)
        chosen.append((text, rank))
        if len(chosen) >= max_items:
            break
    # answers drawn from different filings must say which filing each comes from
    filings = {(retrieved[r - 1].chunk.doc_id) for _, r in chosen}
    out = []
    for text, rank in chosen:
        pre = f"{retrieved[rank - 1].chunk.label.split(' · ')[0]}: " if len(filings) > 1 else ""
        out.append(f"{pre}{text} [{rank}]")
    return "\n".join(out), [r for _, r in chosen]
