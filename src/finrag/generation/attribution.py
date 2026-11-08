"""Fine-grained source attribution + verification.

What is DETERMINISTIC: every citation resolves to a chunk that carries its filing, accession, fiscal period,
Item, note, table id and page as metadata, and the exact supporting span (a table row or sentence).
What is a CHECK (probabilistic, reported as a rate, never promised): whether a generated sentence is actually
supported by the chunk it cites. Support = (a) every number in the sentence appears in the cited source,
(b) enough of the sentence's content words appear there. Both are lexical heuristics; the LLM judge in
finrag.evaluation is the stronger second opinion.

Page numbers: exact for PDFs; for EDGAR HTML they are counted from page-break markers (approximate or absent).
"""
from __future__ import annotations

import re

from finrag.core.numbers import numbers_match, parse_numbers
from finrag.core.schema import Citation, ClaimCheck, RetrievedChunk
from finrag.retrieval.tokenizer import tokenize

_CITE = re.compile(r"\[(\d{1,2})(?:\s*[,;]\s*(\d{1,2}))*\]")
_CITE_NUM = re.compile(r"\d{1,2}")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(“\"•\-\*\d])|\n+")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_ABSTAIN = re.compile(r"^\W*(i\s+(could\s+not|cannot|can['’]t|couldn['’]t|did\s+not)\s+find|the\s+(provided\s+)?(sources|filings|context)\s+(do|does)\s+not|insufficient)", re.I)
SUPPORT_OVERLAP = 0.5


def split_claims(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    return [p for p in parts if len(p) > 2]


def cited_numbers(sentence: str) -> list[int]:
    out: list[int] = []
    for m in _CITE.finditer(sentence):
        out.extend(int(x) for x in _CITE_NUM.findall(m.group(0)))
    return list(dict.fromkeys(out))


def _strip_cites(s: str) -> str:
    return re.sub(r"\s*\[\d{1,2}(?:\s*[,;]\s*\d{1,2})*\]", "", s).strip()


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN.search(text.strip()))


def _units(chunk_text: str) -> list[str]:
    return [ln for ln in chunk_text.splitlines() if ln.strip()]


def best_span(sentence: str, chunk_text: str) -> tuple[str, float]:
    """Best-matching line/sentence inside the chunk, and its token recall for the sentence."""
    stoks = set(tokenize(_strip_cites(sentence)))
    snums = [n for n in parse_numbers(_strip_cites(sentence))]
    best, best_s = "", -1.0
    pieces = _units(chunk_text) if "|" in chunk_text or "\n" in chunk_text else [chunk_text]
    pieces = [p for piece in pieces for p in (_SENT.split(piece) if "|" not in piece else [piece])]
    for piece in pieces:
        ptoks = set(tokenize(piece))
        if not ptoks:
            continue
        rec = len(stoks & ptoks) / (len(stoks) or 1)
        nm = sum(1 for n in snums if any(numbers_match(n, m) for m in parse_numbers(piece)))
        s = rec + 0.6 * nm
        if s > best_s:
            best, best_s = piece.strip(), s
    return best, best_s


def _entity(retrieved: list[RetrievedChunk]) -> set[str]:
    out: set[str] = set()
    for rc in retrieved:
        out |= set(tokenize(rc.chunk.company)) | set(tokenize(rc.chunk.ticker))
    return out


def _check_claim(sentence: str, ncites: list[int], retrieved: list[RetrievedChunk]) -> ClaimCheck:
    body = _strip_cites(sentence)
    if is_abstention(body) or len(tokenize(body)) < 2:
        return ClaimCheck(sentence=sentence, citations=ncites, cited=bool(ncites), supported=None, numeric_ok=None, overlap=0.0)
    issues: list[str] = []
    valid = [n for n in ncites if 1 <= n <= len(retrieved)]
    if len(valid) != len(ncites):
        issues.append("cites a source number that was not provided")
    if not valid:
        issues.append("no citation" if not ncites else "no valid citation")
        return ClaimCheck(sentence=sentence, citations=ncites, cited=False, supported=False, numeric_ok=None, overlap=0.0, issues=issues)

    nums = [n for n in parse_numbers(body)]
    key_nums = [n for n in nums if not _YEAR.match(str(int(abs(n.value)))) or n.scale > 1 or n.pct]
    year_nums = [n for n in nums if n not in key_nums]
    source = "\n".join(f"{retrieved[n - 1].chunk.index_text}" for n in valid)
    src_nums = parse_numbers(source)
    missing = [n.raw for n in key_nums if not any(numbers_match(n, m) for m in src_nums)]
    numeric_ok = None if not key_nums else (not missing)
    if missing:
        issues.append("number not found in cited source: " + ", ".join(missing) + " (computed or wrong?)")
    bad_years = [n.raw for n in year_nums if not any(numbers_match(n, m, scale_tolerant=False) for m in src_nums)]
    if bad_years:
        issues.append("year not in cited source: " + ", ".join(bad_years))

    ent = _entity(retrieved)
    stoks = {t for t in tokenize(body) if not t.isdigit() and t not in ent}
    stoks_src = set(tokenize(source))
    if not nums and len(stoks) < 2:
        issues.append("too little checkable content (company name only?)")
        return ClaimCheck(sentence=sentence, citations=valid, cited=True, supported=False, numeric_ok=None, overlap=0.0, issues=issues)
    overlap = len(stoks & stoks_src) / (len(stoks) or 1)
    # best span across the cited chunks
    span, span_score = "", -1.0
    for n in valid:
        sp, sc = best_span(sentence, retrieved[n - 1].chunk.text)
        if sc > span_score:
            span, span_score = sp, sc
    supported = overlap >= SUPPORT_OVERLAP and (numeric_ok is not False)
    if overlap < SUPPORT_OVERLAP:
        issues.append(f"low word overlap with cited source ({overlap:.0%})")
    return ClaimCheck(sentence=sentence, citations=valid, cited=True, supported=supported, numeric_ok=numeric_ok,
                      overlap=round(overlap, 3), span=span[:400], issues=issues)


def verify_answer(answer: str, retrieved: list[RetrievedChunk]) -> tuple[list[ClaimCheck], dict]:
    if is_abstention(answer):       # a refusal (and its explanation) makes no factual claims to verify
        return ([ClaimCheck(sentence=answer.strip(), citations=[], cited=False, supported=None, numeric_ok=None, overlap=0.0)],
                {"n_claims": 0, "support_rate": None, "citation_coverage": None, "invalid_citations": 0})
    claims = [_check_claim(s, cited_numbers(s), retrieved) for s in split_claims(answer)]
    factual = [c for c in claims if c.supported is not None]
    invalid = sum(1 for s in split_claims(answer) for n in cited_numbers(s) if not 1 <= n <= len(retrieved))
    stats = {
        "n_claims": len(factual),
        "support_rate": (sum(1 for c in factual if c.supported) / len(factual)) if factual else None,
        "citation_coverage": (sum(1 for c in factual if c.cited) / len(factual)) if factual else None,
        "invalid_citations": invalid,
    }
    return claims, stats


def build_citations(claims: list[ClaimCheck], retrieved: list[RetrievedChunk]) -> list[Citation]:
    """One Citation per distinct source actually cited, numbered as in the answer text."""
    spans: dict[int, str] = {}
    for c in claims:
        for n in c.citations:
            if 1 <= n <= len(retrieved) and (n not in spans or c.span):
                spans[n] = c.span or spans.get(n, "")
    out = []
    for n in sorted(spans):
        ch = retrieved[n - 1].chunk
        out.append(Citation(n=n, chunk_id=ch.chunk_id, label=ch.label, ticker=ch.ticker, filing_type=ch.filing_type,
                            fiscal_year=ch.fiscal_year, fiscal_period=ch.fiscal_period, accession=ch.accession, item=ch.item,
                            section=ch.section, note=ch.note, table_id=ch.table_id, page=ch.page, url=ch.url,
                            span=spans[n], element_type=ch.element_type))
    return out
