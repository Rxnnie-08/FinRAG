"""Number handling shared by tokenizer, attribution and evaluation.

Financial text writes the same quantity many ways ("$1.25 billion", "1,250" in a table "in millions",
"(1,250)"). Everything that compares numbers goes through here so they agree.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12, "bn": 1e9, "mm": 1e6}
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}

_NUM_RE = re.compile(
    r"(?P<open>\()?"
    r"(?:(?<![\w])(?P<sign>[-−–]))?\s?"
    r"(?P<cur>[$€£])?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<pct>\s?%)?"
    r"(?:\s?(?P<scale>trillion|billion|million|thousand|bn|mm)\b|(?P<sfx>[kKmMbB])\b)?"
    r"(?P<close>\))?",
    re.I,
)


@dataclass(frozen=True)
class Num:
    value: float          # as written (1.25 in "$1.25 billion")
    scale: float          # 1e9 for "billion", else 1
    pct: bool
    neg: bool
    raw: str

    @property
    def abs_value(self) -> float:
        return abs(self.value) * self.scale


def parse_numbers(text: str) -> list[Num]:
    """Extract numeric mentions. Footnote markers like "(1)" and years' hyphen ranges are not numbers."""
    out: list[Num] = []
    for m in _NUM_RE.finditer(text):
        num = m.group("num")
        opened, closed = bool(m.group("open")), bool(m.group("close"))
        if opened and closed and not m.group("cur") and re.fullmatch(r"\d{1,2}", num):
            prev = text[:m.start()].rstrip()
            if prev and (prev[-1].isalnum() or prev[-1] in ")]"):
                continue  # "... due 2027 (1)": a footnote marker, not a number
            # otherwise "(34)" at the start of a cell / after ':' is a negative value
        value = float(num.replace(",", ""))
        scale = 1.0
        if m.group("scale"):
            scale = _SCALE_WORDS[m.group("scale").lower()]
        elif m.group("sfx"):
            scale = _SUFFIX[m.group("sfx").lower()]
        neg = bool(m.group("sign")) or (opened and closed)
        out.append(Num(value=-value if neg else value, scale=scale, pct=bool(m.group("pct")), neg=neg, raw=m.group(0).strip()))
    return out


def fmt_num(v: float) -> str:
    """Canonical string for a number: 1250.0 -> '1250', 1.25 -> '1.25'."""
    if math.isfinite(v) and abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def numeric_variants(n: Num) -> list[str]:
    """Tokens under which a quantity may appear in a document (used for query-side expansion)."""
    v = abs(n.value)
    toks = {fmt_num(v)}
    if n.scale > 1:
        a = n.abs_value
        for div in (1e3, 1e6, 1e9, 1.0):
            toks.add(fmt_num(a / div))
    return sorted(toks)


def numbers_match(a: Num, b: Num, rel_tol: float = 5e-4, scale_tolerant: bool = True) -> bool:
    """True if two numeric mentions denote the same quantity.

    With scale_tolerant, values equal up to a power of 1000 match (1.25 billion ~ 1,250 million ~ 1,250,000
    thousand). This is deliberately lenient because table units live outside the cell; it can over-match
    (1.25 vs 1,250), so report strict numbers too when the distinction matters.
    """
    av = {abs(a.value), a.abs_value}
    bv = {abs(b.value), b.abs_value}
    factors = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9) if scale_tolerant else (1.0,)
    for x in av:
        for y in bv:
            for f in factors:
                # Cross-scale matches must be exact: a loose tolerance lets 9,999 'match' the 10 in "10-K".
                if math.isclose(x * f, y, rel_tol=rel_tol if f == 1.0 else 1e-9, abs_tol=1e-9):
                    return True
    return False


def any_number_match(pred: str, gold: str, scale_tolerant: bool = True) -> bool:
    gp, pp = parse_numbers(gold), parse_numbers(pred)
    return bool(gp) and any(numbers_match(p, g, scale_tolerant=scale_tolerant) for g in gp for p in pp)
