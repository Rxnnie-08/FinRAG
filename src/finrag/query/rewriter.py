"""Query disambiguation and rewriting.

Pipeline (all deterministic; an LLM pass is optional and only rewrites the wording):
  1. resolve company mentions (ticker, name, alias) against the companies in the corpus,
  2. carry over company / period from the previous turn when the question omits them,
  3. resolve period phrases against the company's fiscal calendar ("Q2 2025", "last quarter", "FY24"),
  4. spell out abbreviations and collect BM25 synonyms,
  5. emit metadata filters + a list of the assumptions made (shown to the user, never hidden).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from finrag.core.fiscal import KNOWN_FYE, PeriodSpec, parse_period
from finrag.core.schema import FilingMeta, RewrittenQuery

from .glossary import expand_terms

_SUFFIXES = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|ltd|plc|holdings|group|llc)\b\.?", re.I)
_PRONOUN = re.compile(r"\b(they|them|their|its|it|the company|the firm)\b", re.I)
_POSSESSIVE = re.compile(r"(?<![\w])((?:[A-Z][A-Za-z&.\-]+)(?:\s+[A-Z][A-Za-z&.\-]+){0,3})(?:’s|'s|’|')(?![A-Za-z])")
_CORP = re.compile(r"((?:[A-Z][A-Za-z&.\-]+\s+){1,3}(?:Inc|Corp|Corporation|Company|Co|Ltd|plc|Holdings|Group|LLC)\b\.?)")
_NOT_ENTITY = {"q1", "q2", "q3", "q4", "fy", "gaap", "sec", "item", "note", "table", "sofr", "ebitda", "ebit", "eps", "january", "february",
               "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "what", "how", "which",
               "who", "when", "where", "why", "did", "does", "is", "are", "was", "were", "term", "senior", "notes", "loan", "company", "10-k", "10-q", "8-k"}
_FOLLOWUP = re.compile(r"^(and|what about|how about|also|then|same for|compare)\b", re.I)


# Nicknames people use in questions, keyed by ticker (the FinanceBench set uses AMEX, JnJ, JPM, ...).
EXTRA_ALIASES = {
    "AMERICANEXPRESS": ["amex"], "JOHNSON_JOHNSON": ["jnj", "j&j"], "JPMORGAN": ["jpm", "jp morgan", "jpmorgan chase"],
    "COCACOLA": ["coke"], "GENERALMILLS": ["gis"], "BESTBUY": ["bby"], "AMAZON": ["amzn"], "MICROSOFT": ["msft"],
    "ACTIVISIONBLIZZARD": ["activision"], "AMERICANWATERWORKS": ["american water", "awk"], "PEPSICO": ["pepsi"],
    "PAYPAL": ["pypl"], "LOCKHEEDMARTIN": ["lockheed"], "MGMRESORTS": ["mgm"], "ULTABEAUTY": ["ulta"], "FOOTLOCKER": ["foot locker"],
}


@dataclass
class CompanyDirectory:
    """Companies present in the corpus: ticker -> {name, fye, aliases}."""
    entries: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_filings(cls, filings: list[FilingMeta]) -> "CompanyDirectory":
        d: dict[str, dict] = {}
        for f in filings:
            e = d.setdefault(f.ticker.upper(), {"name": f.company, "fye": f.fiscal_year_end or KNOWN_FYE.get(f.ticker.upper(), "1231"), "aliases": set(), "years": set()})
            if f.fiscal_year:
                e["years"].add(int(f.fiscal_year))
            core = _SUFFIXES.sub("", f.company).strip(" .,")
            e["aliases"].update({f.company.lower(), core.lower(), core.split()[0].lower() if core else ""})
            e["aliases"].update(core.lower().replace("-", " ").replace("&", " & ") and {core.lower().replace("-", " ")})
            e["aliases"].update(EXTRA_ALIASES.get(f.ticker.upper(), []))
        for e in d.values():
            e["aliases"] = {a for a in e["aliases"] if len(a) >= 3}
        return cls(d)

    @classmethod
    def from_chunks(cls, chunks) -> "CompanyDirectory":
        seen: dict[str, FilingMeta] = {}
        for c in chunks:
            seen.setdefault((c.ticker, c.fiscal_year), FilingMeta(company=c.company, ticker=c.ticker, filing_type=c.filing_type,
                            period_end=c.period_end, fiscal_year=c.fiscal_year, fiscal_year_end=KNOWN_FYE.get(c.ticker.upper(), "1231")))
        return cls.from_filings(list(seen.values()))

    def find(self, text: str) -> list[str]:
        found: list[tuple[int, str]] = []
        low = text.lower()
        for t, e in self.entries.items():
            pos = None
            m = re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", text)       # tickers are case-sensitive
            if m:
                pos = m.start()
            else:
                for a in sorted(e["aliases"], key=len, reverse=True):
                    m2 = re.search(rf"(?<![a-z]){re.escape(a)}(?:’s|'s)?(?![a-z])", low)
                    if m2:
                        pos = m2.start()
                        break
            if pos is not None:
                found.append((pos, t))
        return [t for _, t in sorted(found)]

    def years(self, tickers: list[str]) -> set[int]:
        return {y for t in tickers for y in self.entries.get(t, {}).get("years", set())}

    def fye(self, ticker: str) -> str:
        return self.entries.get(ticker, {}).get("fye", "1231")


class QueryRewriter:
    def __init__(self, directory: CompanyDirectory, llm=None, today: date | None = None, mode: str = "rules"):
        if mode not in ("off", "rules", "llm"):
            raise ValueError("mode must be off | rules | llm")
        self.dir, self.llm, self.today, self.mode = directory, llm, today, mode

    def rewrite(self, question: str, history: list[str] | None = None) -> RewrittenQuery:
        q = question.strip()
        if self.mode == "off":
            return RewrittenQuery(original=q, standalone=q, sparse_query=q, mode="off")

        assumptions: list[str] = []
        clarification = ""
        history_q = history[-1] if history else ""
        tickers = self.dir.find(q)

        carried = False
        if not tickers and history_q:
            prev = self.dir.find(history_q)
            if prev and (_PRONOUN.search(q) or _FOLLOWUP.search(q) or len(q.split()) <= 9):
                tickers, carried = prev, True
                assumptions.append(f"Company not named; carried over {', '.join(tickers)} from the previous question.")
        if not tickers:
            if len(self.dir.entries) == 1:
                tickers = list(self.dir.entries)
                assumptions.append(f"Only one company in the corpus; assumed {tickers[0]}.")
            elif len(self.dir.entries) > 1:
                clarification = "Which company do you mean? Searching all companies; name one (e.g. " + ", ".join(list(self.dir.entries)[:3]) + ") for a precise answer."

        unknown = []
        if not tickers:
            for rx in (_POSSESSIVE, _CORP):
                for m in rx.finditer(q):
                    name = m.group(1).strip(" .")
                    if name.lower() in _NOT_ENTITY or name.split()[0].lower() in _NOT_ENTITY or name in unknown:
                        continue
                    unknown.append(name)
            if unknown and not (carried or (history_q and self.dir.find(history_q) and _PRONOUN.search(q))):
                avail = ", ".join(self.dir.entries)
                assumptions.append(f"“{unknown[0]}” is not a company in the corpus (available: {avail}).")
                clarification = ""
            else:
                unknown = []

        fye_set = {self.dir.fye(t) for t in tickers}
        fye = next(iter(fye_set)) if fye_set else "1231"
        if len(fye_set) > 1:
            assumptions.append("Companies have different fiscal year ends; a quarter or year is matched to each company's own fiscal calendar, not the calendar year.")
        spec: PeriodSpec = parse_period(q, fye, self.today or date.today())
        if not spec.matched and history_q and carried:
            hspec = parse_period(history_q, fye, self.today or date.today())
            if hspec.matched:
                spec = hspec
                assumptions.append(f"Period not stated; carried over {', '.join(hspec.matched)} from the previous question.")
        if spec.bare_year and spec.fiscal_year:
            corpus_years = self.dir.years(tickers or list(self.dir.entries))
            m = re.search(r"\b(due|matur\w*|through|until|beyond|by|after|expir\w*)\b[^.?]{0,25}\b" + str(spec.fiscal_year), q, re.I)
            if m or (corpus_years and spec.fiscal_year not in corpus_years):
                why = "it follows a maturity/date cue" if m else f"no filing for fiscal {spec.fiscal_year} is in the corpus"
                note = f"{spec.fiscal_year} treated as a date inside the question, not a filing period ({why})."
                spec = PeriodSpec(filing_type=spec.filing_type, assumptions=[note])
        assumptions.extend(a for a in spec.assumptions if a not in assumptions)
        if tickers and not (spec.fiscal_year or spec.quarter or spec.latest or spec.annual) and spec.filing_type != "8-K":
            assumptions.append("No period stated: all fiscal periods in the corpus are searched. Name a fiscal year or quarter for a single answer.")

        filters: dict = {}
        if tickers:
            filters["ticker"] = tickers
        ftype = spec.filing_type
        if spec.quarter == 4:
            ftype = ftype or "10-K"
        elif spec.quarter and not ftype:
            ftype = "10-Q"
        if ftype:
            filters["filing_type"] = [ftype]
        if ftype != "8-K":
            if spec.fiscal_year:
                filters["fiscal_year"] = [spec.fiscal_year]
            if spec.quarter == 4:
                filters["fiscal_period"] = ["FY"]
            elif spec.quarter:
                filters["fiscal_period"] = [f"Q{spec.quarter}"]
            elif spec.annual and not ftype:
                filters["fiscal_period"] = ["FY"]
            if spec.latest:
                filters["latest"] = True

        standalone, synonyms, notes = expand_terms(q)
        if carried and tickers:
            names = ", ".join(self.dir.entries[t]["name"] for t in tickers)
            if not any(self.dir.find(standalone)):
                standalone = f"{standalone.rstrip('?')} ({names})?"
        expansions = notes
        used_llm = False
        if self.mode == "llm" and self.llm is not None:
            try:
                standalone = self._llm_rewrite(standalone, history_q, tickers) or standalone
                used_llm = True
            except Exception as e:  # network / key problems must not break retrieval
                assumptions.append(f"LLM rewrite failed ({type(e).__name__}); used rule-based rewrite.")
        sparse = " ".join([standalone] + synonyms)
        return RewrittenQuery(original=q, standalone=standalone, sparse_query=sparse, filters=filters, assumptions=assumptions,
                              clarification=clarification, expansions=expansions, mode=self.mode, used_llm=used_llm,
                              unknown_entities=unknown)

    def _llm_rewrite(self, standalone: str, history_q: str, tickers: list[str]) -> str:
        system = ("Rewrite the user's question about SEC filings as one self-contained question. Keep every number, company, "
                  "and period exactly as given; do not answer it; do not add facts. Reply with JSON: {\"question\": \"...\"}.")
        user = json.dumps({"previous_question": history_q, "question": standalone, "known_companies": tickers})
        out = self.llm.complete(system, user, max_tokens=200)
        m = re.search(r"\{.*\}", out, re.S)
        text = json.loads(m.group(0))["question"] if m else out
        text = text.strip()
        return text if 5 < len(text) < 600 else standalone
