"""SEC EDGAR client (submissions API + document download).

SEC fair-access rules: identify yourself in the User-Agent (name + email) and stay under 10 requests/second.
Set SEC_USER_AGENT="Your Name your.email@university.edu". Nothing is hard-coded here on purpose.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import requests

from finrag.core.fiscal import fiscal_period_for_date, parse_fye
from finrag.core.schema import FilingMeta

DATA_BASE = "https://data.sec.gov"
WWW_BASE = "https://www.sec.gov"


class EdgarClient:
    def __init__(self, user_agent: str | None, session: requests.Session | None = None, min_interval: float = 0.2):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with contact details, e.g. 'Jane Doe jane@university.edu'. "
                "Set the SEC_USER_AGENT environment variable.")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self._last = 0.0
        self._tickers: dict | None = None

    def _get(self, url: str) -> requests.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=60)
        self._last = time.monotonic()
        resp.raise_for_status()
        return resp

    def resolve(self, ticker: str) -> tuple[str, str]:
        """ticker -> (10-digit CIK, company name)."""
        if self._tickers is None:
            self._tickers = self._get(f"{WWW_BASE}/files/company_tickers.json").json()
        t = ticker.upper()
        for row in self._tickers.values():
            if row["ticker"].upper() == t:
                return str(row["cik_str"]).zfill(10), row["title"]
        raise KeyError(f"Ticker not found on EDGAR: {ticker}")

    def list_filings(self, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 4) -> list[FilingMeta]:
        cik, name = self.resolve(ticker)
        sub = self._get(f"{DATA_BASE}/submissions/CIK{cik}.json").json()
        fye = sub.get("fiscalYearEnd") or "1231"
        fye_month, _ = parse_fye(fye)
        recent = sub["filings"]["recent"]
        out: list[FilingMeta] = []
        for i, form in enumerate(recent["form"]):
            if form not in forms:
                continue
            report = recent["reportDate"][i] or recent["filingDate"][i]
            acc = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            fy, q = (None, 0)
            period = ""
            if form in ("10-K", "10-Q"):
                fy, q = fiscal_period_for_date(date.fromisoformat(report), fye_month)
                period = "FY" if form == "10-K" else f"Q{q}"
            out.append(FilingMeta(
                company=sub.get("name", name), ticker=ticker.upper(), cik=cik, filing_type=form, period_end=report,
                accession=acc, fiscal_year=fy, fiscal_period=period, fiscal_year_end=fye,
                filing_date=recent["filingDate"][i],
                url=f"{WWW_BASE}/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"))
        out.sort(key=lambda m: m.period_end, reverse=True)
        return out[:limit]   # note: only the 'recent' window (~1000 filings) is covered

    def download(self, meta: FilingMeta, dest_dir: str | Path) -> FilingMeta:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        ext = Path(meta.url).suffix or ".htm"
        path = dest / f"{meta.ticker}_{meta.filing_type}_{meta.period_end}{ext}"
        path.write_bytes(self._get(meta.url).content)
        meta.path = str(path)
        return meta
