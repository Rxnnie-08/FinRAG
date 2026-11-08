"""Inline-XBRL fact extraction.

Recent 10-K/10-Q HTML files tag financial-statement numbers (`ix:nonFraction`). Reading those tags gives
exact, machine-verified values for the headline line items, so the balance sheet / income statement can be
answered without trusting a parsed table. Only consolidated (dimension-free) facts are used.
"""
from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup

from finrag.core.numbers import fmt_num
from finrag.core.schema import Chunk, FilingMeta

KEY_CONCEPTS: dict[str, str] = {
    "Assets": "Total assets",
    "AssetsCurrent": "Total current assets",
    "Liabilities": "Total liabilities",
    "LiabilitiesCurrent": "Total current liabilities",
    "StockholdersEquity": "Total stockholders' equity",
    "LiabilitiesAndStockholdersEquity": "Total liabilities and stockholders' equity",
    "CashAndCashEquivalentsAtCarryingValue": "Cash and cash equivalents",
    "AccountsReceivableNetCurrent": "Accounts receivable, net",
    "InventoryNet": "Inventories, net",
    "LongTermDebt": "Total long-term debt",
    "LongTermDebtNoncurrent": "Long-term debt, net of current portion",
    "OperatingLeaseLiability": "Operating lease liabilities",
    "FinanceLeaseLiability": "Finance lease liabilities",
    "OperatingLeaseRightOfUseAsset": "Operating lease right-of-use assets",
    "Revenues": "Total revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
    "CostOfRevenue": "Cost of revenue",
    "GrossProfit": "Gross profit",
    "OperatingIncomeLoss": "Operating income",
    "NetIncomeLoss": "Net income",
    "EarningsPerShareBasic": "Basic earnings per share",
    "EarningsPerShareDiluted": "Diluted earnings per share",
    "NetCashProvidedByUsedInOperatingActivities": "Net cash from operating activities",
    "PaymentsToAcquirePropertyPlantAndEquipment": "Capital expenditures",
}


@dataclass
class Fact:
    concept: str
    value: float
    unit: str
    instant: str = ""
    start: str = ""
    end: str = ""
    display: str = ""

    @property
    def period(self) -> str:
        return self.instant or (f"{self.start} to {self.end}" if self.start else self.end)


def _contexts(soup: BeautifulSoup) -> dict[str, dict]:
    ctx: dict[str, dict] = {}
    for c in soup.find_all("xbrli:context"):
        cid = c.get("id")
        if not cid:
            continue
        has_dims = bool(c.find(["xbrldi:explicitmember", "xbrldi:typedmember"]))
        inst = c.find("xbrli:instant")
        st, en = c.find("xbrli:startdate"), c.find("xbrli:enddate")
        ctx[cid] = {
            "dims": has_dims,
            "instant": inst.get_text(strip=True) if inst else "",
            "start": st.get_text(strip=True) if st else "",
            "end": en.get_text(strip=True) if en else "",
        }
    return ctx


def extract_facts(html: str) -> list[Fact]:
    if "ix:nonfraction" not in html.lower():
        return []
    soup = BeautifulSoup(html, "lxml")
    ctx = _contexts(soup)
    facts: list[Fact] = []
    for tag in soup.find_all("ix:nonfraction"):
        name = tag.get("name", "")
        concept = name.split(":")[-1]
        c = ctx.get(tag.get("contextref", ""))
        if not c or c["dims"]:
            continue
        raw = tag.get_text(strip=True).replace(",", "")
        if tag.get("format", "").endswith(("zerodash", "fixed-zero")) or raw in ("-", "—", "–"):
            val = 0.0
        else:
            try:
                val = float(raw)
            except ValueError:
                continue
        try:
            val *= 10 ** int(tag.get("scale", "0") or 0)
        except ValueError:
            pass
        if tag.get("sign") == "-":
            val = -val
        facts.append(Fact(concept=concept, value=val, unit=(tag.get("unitref") or "").lower(), instant=c["instant"],
                          start=c["start"], end=c["end"], display=tag.get_text(strip=True)))
    return facts


def _fmt_usd(f: Fact) -> str:
    if f.unit and "usd" in f.unit and "share" not in f.unit:
        v = f.value
        if abs(v) >= 1e6:
            m = v / 1e6
            return f"${m:,.0f} million" if abs(m - round(m)) < 0.05 else f"${m:,.1f} million"
        return f"${v:,.2f}"
    return fmt_num(f.value) + (f" {f.unit}" if f.unit else "")


def xbrl_chunks(html: str, meta: FilingMeta, start_seq: int = 0) -> list[Chunk]:
    facts = [f for f in extract_facts(html) if f.concept in KEY_CONCEPTS]
    if not facts:
        return []
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for f in sorted(facts, key=lambda x: (list(KEY_CONCEPTS).index(x.concept), x.period), reverse=False):
        k = (f.concept, f.period)
        if k in seen:
            continue
        seen.add(k)
        kind = "as of" if f.instant else "for the period"
        lines.append(f"{KEY_CONCEPTS[f.concept]} (us-gaap:{f.concept}) {kind} {f.period}: {_fmt_usd(f)}")
    text = "Key financial facts (XBRL-tagged, consolidated)\n" + "\n".join(lines)
    return [Chunk(
        chunk_id=f"{meta.doc_id}#x{start_seq + 1:04d}", text=text, company=meta.company, ticker=meta.ticker,
        filing_type=meta.filing_type, fiscal_year=meta.fiscal_year, fiscal_period=meta.fiscal_period,
        period_end=meta.period_end, accession=meta.accession, source=meta.path, item="Item 8",
        section="Financial Statements (XBRL facts)", element_type="xbrl_facts", url=meta.url)]
