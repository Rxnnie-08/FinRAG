"""Number-preserving tokenizer for BM25.

A default text tokenizer destroys exactly what financial questions are about: "$5,240" becomes "5" and "240",
"4.250%" becomes "4" and "250". Here a number stays one token in canonical form ("5240", "4.25"), so a query
for 5,240 matches only 5,240. On the query side, numbers written with a scale word ("$5.24 billion") also
emit the forms a table might use ("5240" in millions), so BM25 can find them.
"""
from __future__ import annotations

import re

from finrag.core.numbers import fmt_num, numeric_variants, parse_numbers

STOPWORDS = frozenset(
    "a an and are as at be been but by for from has have had in into is it its of on or that the their this to was were "
    "what which who whom when where how did do does would will with than then there these those during after before "
    "over under per not no we our us they them his her he she you your i".split())

_WORD = re.compile(r"[a-z][a-z0-9'’\-]*|\d[\d,]*(?:\.\d+)?%?")
_NUM_TOKEN = re.compile(r"^\d[\d,]*(?:\.\d+)?%?$")


def _norm_number(tok: str) -> str:
    t = tok.rstrip("%").replace(",", "")
    try:
        return fmt_num(float(t))
    except ValueError:
        return t


def _stem(w: str) -> str:
    """Very light plural/'ing' stemming: enough to match 'leases'~'lease', 'maturities'~'maturity'."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for m in _WORD.finditer(text.lower().replace("’", "'")):
        tok = m.group(0)
        if _NUM_TOKEN.match(tok):
            out.append(_norm_number(tok))
            continue
        tok = tok.strip("-'")
        if tok.endswith("'s"):
            tok = tok[:-2]
        if not tok or (not keep_stopwords and tok in STOPWORDS):
            continue
        out.append(_stem(tok))
        # split hyphen compounds so "right-of-use" also matches "right of use"
        if "-" in tok:
            out.extend(_stem(p) for p in tok.split("-") if p and p not in STOPWORDS)
    return out


def tokenize_query(text: str) -> list[str]:
    """Query tokens, plus scale variants for numbers ('1.25 billion' -> also '1250', '1250000')."""
    toks = tokenize(text)
    extra: list[str] = []
    for n in parse_numbers(text):
        extra.extend(numeric_variants(n))
    seen = set(toks)
    return toks + [e for e in extra if e not in seen]
