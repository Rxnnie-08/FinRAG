"""Fiscal calendars and period parsing.

"Q3 2024", "last quarter" and "fiscal 2023" mean different dates for different companies. Everything is
resolved against the company's fiscal year end (EDGAR exposes it as `fiscalYearEnd`, MMDD).

Conventions: a fiscal year is named for the calendar year in which it ends ("end", used by Apple, Microsoft,
Nvidia, ...). Some retailers name it for the year in which it starts ("start"); pass convention="start".
52/53-week calendars are handled at month granularity (a period ending Jan 2 is treated as ending in December).
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

# Month-level approximations for convenience when offline. EDGAR's fiscalYearEnd always wins.
KNOWN_FYE = {
    "AAPL": "0930", "MSFT": "0630", "TSLA": "1231", "AMZN": "1231", "GOOGL": "1231",
    "GOOG": "1231", "META": "1231", "NVDA": "0131", "WMT": "0131", "ORCL": "0531",
}

_ORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def parse_fye(fye: str | None) -> tuple[int, int]:
    """'0930' -> (9, 30). Unknown -> December 31."""
    if not fye or not re.fullmatch(r"\d{4}", fye):
        return 12, 31
    m, d = int(fye[:2]), int(fye[2:])
    return (m, d) if 1 <= m <= 12 and 1 <= d <= 31 else (12, 31)


def month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def fiscal_period_for_date(end: date, fye_month: int) -> tuple[int, int]:
    """(fiscal_year, quarter 1-4) for a period ending on `end`."""
    eff = end - timedelta(days=7) if end.day <= 7 else end   # 52/53-week calendars
    fy = eff.year if eff.month <= fye_month else eff.year + 1
    diff = (fye_month - eff.month) % 12
    q = 4 - int(round(diff / 3))
    return fy, min(4, max(1, q))


def quarter_end(fye_month: int, fy: int, q: int, convention: str = "end") -> date:
    m = fye_month - 3 * (4 - q)
    y = fy if not (convention == "start" and fye_month < 12) else fy + 1
    while m <= 0:
        m += 12
        y -= 1
    return month_end(y, m)


def latest_completed_quarter(today: date, fye_month: int, convention: str = "end") -> tuple[int, int]:
    for fy in range(today.year + 1, today.year - 4, -1):
        for q in (4, 3, 2, 1):
            if quarter_end(fye_month, fy, q, convention) < today:
                return fy, q
    raise ValueError("no completed quarter found")


def latest_completed_fy(today: date, fye_month: int, convention: str = "end") -> int:
    fy, q = latest_completed_quarter(today, fye_month, convention)
    return fy if q == 4 else fy - 1


@dataclass
class PeriodSpec:
    fiscal_year: int | None = None
    quarter: int | None = None
    annual: bool = False
    latest: bool = False
    filing_type: str | None = None
    assumptions: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    bare_year: bool = False     # year found with no 'fiscal'/'FY'/quarter cue: may not be a filing period

    @property
    def fiscal_period(self) -> str:
        if self.quarter:
            return f"Q{self.quarter}"
        return "FY" if self.annual else ""


def _norm_years(t: str) -> str:
    t = re.sub(r"\bFY\s*[’']?\s*(\d{2})\b(?!\d)", r"FY20\1", t, flags=re.I)
    t = re.sub(r"\b(Q[1-4])\s*[’']\s*(\d{2})\b", r"\1 20\2", t, flags=re.I)
    t = re.sub(r"\bfiscal\s+(\d{2})\b(?!\d)", r"fiscal 20\1", t, flags=re.I)
    return t


_I = re.I
_ENDED = re.compile(r"\b(year|quarter|three months|six months|nine months)s?\s+ended\s+([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})", _I)
_Q_YR = re.compile(r"\bQ([1-4])\s*(?:of\s+)?(?:FY|fiscal(?:\s+year)?)?\s*(\d{4})\b", _I)
_Q_ORD = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:fiscal\s+)?quarter(?:\s+of)?\s*(?:fiscal\s+|FY\s*)?(?:year\s+)?(\d{4})?", _I)
_FY = re.compile(r"\b(?:FY|fiscal(?:\s+year)?)\s*(\d{4})\b", _I)
_REL_Q = re.compile(r"\b(?:last|latest|most recent|previous|prior)\s+(?:fiscal\s+)?quarter\b", _I)
_REL_Y_PAST = re.compile(r"\b(?:last|previous|prior)\s+(?:fiscal\s+)?year\b", _I)
_REL_Y_NOW = re.compile(r"\b(?:this|current)\s+(?:fiscal\s+)?year\b|\b(?:latest|most recent)\s+(?:fiscal\s+)?(?:year|annual report)\b", _I)
_Q_ONLY = re.compile(r"\bQ([1-4])\b", _I)
_BARE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def detect_filing_type(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b10[- ]?k\b|annual report", t):
        return "10-K"
    if re.search(r"\b10[- ]?q\b|quarterly report", t):
        return "10-Q"
    if re.search(r"\b8[- ]?k\b|press release|current report|earnings release", t):
        return "8-K"
    return None


def parse_period(text: str, fye: str | None = "1231", today: date | None = None, convention: str = "end") -> PeriodSpec:
    """Resolve period references in a question against a fiscal calendar."""
    today = today or date.today()
    fye_month, _ = parse_fye(fye)
    t = _norm_years(text)
    spec = PeriodSpec(filing_type=detect_filing_type(t))
    fye_txt = f"fiscal year ends {calendar.month_name[fye_month]}" if fye_month != 12 else "calendar fiscal year"

    if m := _ENDED.search(t):
        kind, mon, day, yr = m.group(1).lower(), m.group(2).lower(), int(m.group(3)), int(m.group(4))
        if mon in _MONTHS:
            end = date(yr, _MONTHS[mon], min(day, calendar.monthrange(yr, _MONTHS[mon])[1]))
            fy, q = fiscal_period_for_date(end, fye_month)
            spec.fiscal_year = fy
            if kind == "year":
                spec.annual = True
            else:
                spec.quarter = q
            spec.matched.append(m.group(0))
            spec.assumptions.append(f"Period ending {end.isoformat()} read as fiscal {fy}" + ("" if kind == "year" else f" Q{q}") + f" ({fye_txt}).")
            return spec

    if m := _Q_YR.search(t):
        spec.quarter, spec.fiscal_year = int(m.group(1)), int(m.group(2))
        spec.matched.append(m.group(0))
    elif m := _Q_ORD.search(t):
        spec.quarter = _ORD[m.group(1).lower()]
        if m.group(2):
            spec.fiscal_year = int(m.group(2))
        spec.matched.append(m.group(0))
    elif m := _FY.search(t):
        spec.fiscal_year, spec.annual = int(m.group(1)), True
        spec.matched.append(m.group(0))
    elif m := _REL_Q.search(t):
        fy, q = latest_completed_quarter(today, fye_month, convention)
        spec.fiscal_year, spec.quarter = fy, q
        spec.matched.append(m.group(0))
        spec.assumptions.append(f"'{m.group(0)}' read as the last completed fiscal quarter as of {today.isoformat()}: fiscal {fy} Q{q} ({fye_txt}). The filing may not be in the corpus yet.")
    elif m := _REL_Y_PAST.search(t):
        fy = latest_completed_fy(today, fye_month, convention)
        spec.fiscal_year, spec.annual = fy, True
        spec.matched.append(m.group(0))
        spec.assumptions.append(f"'{m.group(0)}' read as the most recently completed fiscal year: fiscal {fy} ({fye_txt}).")
    elif m := _REL_Y_NOW.search(t):
        spec.latest, spec.annual = True, True
        spec.matched.append(m.group(0))
        spec.assumptions.append(f"'{m.group(0)}' read as the latest filing available in the corpus.")
    elif m := _Q_ONLY.search(t):
        spec.quarter = int(m.group(1))
        spec.matched.append(m.group(0))
        spec.assumptions.append(f"Q{spec.quarter} given without a year; all years are searched.")
    elif m := _BARE_YEAR.search(t):
        spec.fiscal_year = int(m.group(1))
        spec.bare_year = True
        spec.matched.append(m.group(0))
        spec.assumptions.append(f"Year {spec.fiscal_year} read as fiscal year {spec.fiscal_year}" + (f" ({fye_txt}), which differs from the calendar year." if fye_month != 12 else "."))

    if spec.quarter == 4:
        spec.assumptions.append("Q4 results are reported in the 10-K, not a 10-Q.")
    if spec.quarter and spec.fiscal_year and fye_month != 12:
        end = quarter_end(fye_month, spec.fiscal_year, spec.quarter, convention)
        spec.assumptions.append(f"Fiscal {spec.fiscal_year} Q{spec.quarter} ends {end.isoformat()} ({fye_txt}).")
    return spec
