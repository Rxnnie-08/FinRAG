"""Work out which company / form / period a filing is from the filing itself, so users can just upload a file.

Order of trust:
  1. inline-XBRL cover tags (`dei:*`) in modern EDGAR HTML: exact,
  2. the cover-page text ("FORM 10-K", "for the fiscal year ended September 28, 2024", "(Exact name of registrant ...)"),
  3. the file name (AAPL_10-K_2024.htm).
Anything the user types in the upload form overrides detection. Every guess that could be wrong is reported in `notes`.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup

from finrag.core.fiscal import KNOWN_FYE
from finrag.core.schema import FilingMeta

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Sept|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|Jun|Jul|Aug"
_DATE = rf"((?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}})"
_PERIOD = re.compile(rf"for\s+the\s+(?:fiscal\s+year|quarterly\s+period|transition\s+period|year|period)\s+ended[:\s]+{_DATE}", re.I)
_FORM = re.compile(r"\bform\s+(10-?K|10-?Q|8-?K)\b", re.I)
_REGISTRANT = re.compile(r"\(\s*exact\s+name\s+of\s+registrant", re.I)


def _parse_date(s: str) -> date | None:
    s = re.sub(r"\s+", " ", s.replace(".", "").replace(",", "")).strip()
    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(re.sub(r"\bSept\b", "Sep", s), fmt).date()
        except ValueError:
            pass
    return None


def _dei(html: str) -> dict[str, str]:
    if "dei:" not in html.lower():
        return {}
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, str] = {}
    for tag in soup.find_all(["ix:nonnumeric", "ix:nonfraction"]):
        name = (tag.get("name") or "")
        if name.lower().startswith("dei:"):
            out.setdefault(name.split(":")[1], tag.get_text(" ", strip=True))
    return out


def _front_text(path: Path, html: str | None) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("PDF filings need pypdf: pip install pypdf") from e
        r = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in r.pages[:3])
    soup = BeautifulSoup(html or "", "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    return soup.get_text("\n")[:20000]


def _company_from_text(text: str) -> str:
    bad = r"commission file|^[\d\-\s().]+$|state or other|\(|^form\b|table of contents|^part\b|^item\b|securities and exchange"
    m = _REGISTRANT.search(text)
    if m:
        lines = [ln.strip() for ln in text[:m.start()].splitlines() if ln.strip()]
        for ln in reversed(lines[-4:]):
            if 2 < len(ln) < 80 and not re.search(bad, ln, re.I):
                return ln.strip(" ,")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # fallback 1: the line after "Commission file number ..." on the cover
    for i, ln in enumerate(lines[:60]):
        if re.match(r"commission\s+file\s+number", ln, re.I) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if 2 < len(nxt) < 60 and not re.search(bad, nxt, re.I):
                return nxt.strip(" ,")
    # fallback 2: the line right after the "FORM 10-K ..." line on the cover
    for i, ln in enumerate(lines[:40]):
        if _FORM.search(ln) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if 2 < len(nxt) < 60 and not re.search(bad, nxt, re.I):
                return nxt.strip(" ,")
    return ""


def _from_filename(name: str) -> dict:
    stem = Path(name).stem
    out: dict = {}
    m = re.search(r"10-?K|10-?Q|8-?K", stem, re.I)
    if m:
        out["filing_type"] = m.group(0).upper().replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K")
    t = re.match(r"([A-Za-z0-9]{1,6})[_\- ]", stem)
    if t:
        out["ticker"] = t.group(1).upper()
    d = re.search(r"(\d{4})-(\d{2})-(\d{2})", stem)
    if d:
        out["period_end"] = d.group(0)
    return out


def detect_meta(path: str | Path, overrides: dict | None = None, original_name: str | None = None) -> tuple[FilingMeta, list[str]]:
    """Returns (FilingMeta, notes). Raises ValueError when form type or period cannot be determined and were not supplied."""
    path = Path(path)
    ov = {k: v.strip() for k, v in (overrides or {}).items() if isinstance(v, str) and v.strip()}
    notes: list[str] = []
    html = path.read_text(encoding="utf-8", errors="ignore") if path.suffix.lower() != ".pdf" else None
    dei = _dei(html) if html else {}
    text = _front_text(path, html)
    fn = _from_filename(original_name or path.name)

    ftype = ov.get("filing_type") or ""
    if not ftype and dei.get("DocumentType"):
        ftype = dei["DocumentType"].upper().split("/")[0]
    if not ftype:
        m = _FORM.search(text[:6000])
        ftype = m.group(1).upper().replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K") if m else fn.get("filing_type", "")
    ftype = ftype.upper().replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K")
    if ftype not in ("10-K", "10-Q", "8-K"):
        raise ValueError("Could not tell whether this is a 10-K, 10-Q or 8-K. Choose the form type in the details box.")

    pe = ov.get("period_end") or dei.get("DocumentPeriodEndDate") or ""
    d = _parse_date(pe) if pe and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", pe) else (date.fromisoformat(pe) if pe else None)
    if d is None:
        m = _PERIOD.search(text[:8000])
        d = _parse_date(m.group(1)) if m else None
    if d is None and fn.get("period_end"):
        d = date.fromisoformat(fn["period_end"])
    if d is None:
        raise ValueError("Could not find the period end date. Enter it (YYYY-MM-DD) in the details box.")

    company = ov.get("company") or dei.get("EntityRegistrantName") or _company_from_text(text)
    ticker = (ov.get("ticker") or dei.get("TradingSymbol") or fn.get("ticker") or "").upper()
    if not company:
        company = ticker or path.stem
        notes.append("Company name not found; using the file name. Enter it in the details box for better matching.")
    if not ticker:
        ticker = re.sub(r"[^A-Za-z0-9]", "", company.split()[0]).upper()[:6] or "DOC"
        notes.append(f"Ticker not found; using {ticker}.")

    fye = ov.get("fiscal_year_end") or ""
    if not fye and dei.get("CurrentFiscalYearEndDate"):
        m = re.search(r"(\d{2})-(\d{2})", dei["CurrentFiscalYearEndDate"])
        fye = (m.group(1) + m.group(2)) if m else ""
    if not fye:
        if ftype == "10-K":
            fye = f"{d.month:02d}{d.day:02d}"
        elif ticker in KNOWN_FYE:
            fye = KNOWN_FYE[ticker]
        else:
            fye = "1231"
            notes.append("Fiscal year end unknown for this 10-Q/8-K; assumed December 31. Quarter questions may resolve wrongly if that is not right.")

    meta = FilingMeta(company=company, ticker=ticker, filing_type=ftype, period_end=d.isoformat(), fiscal_year_end=fye,
                      accession=f"UPLOAD-{path.stem}", cik=dei.get("EntityCentralIndexKey", ""), path=str(path))
    if dei.get("DocumentFiscalYearFocus", "").isdigit():
        meta.fiscal_year = int(dei["DocumentFiscalYearFocus"])
    fp = dei.get("DocumentFiscalPeriodFocus", "").upper()
    if fp in ("FY", "Q1", "Q2", "Q3"):
        meta.fiscal_period = fp
    return meta, notes
