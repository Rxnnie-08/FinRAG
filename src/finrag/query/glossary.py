"""Financial terminology expansion. Maps abbreviations and informal terms to the wording filings actually use.

`abbrev` entries are spelled out in the standalone question (readable by the dense retriever and the LLM);
`synonyms` are appended to the BM25 query only, so they widen keyword recall without changing the question.
"""
from __future__ import annotations

import re

# term (regex, matched case-insensitively on word boundaries) -> (expansion words, kind)
_ENTRIES: list[tuple[str, list[str], str]] = [
    (r"capex", ["capital expenditures", "purchases of property plant and equipment"], "abbrev"),
    (r"cap\s?ex", ["capital expenditures"], "abbrev"),
    (r"eps", ["earnings per share", "diluted", "basic"], "abbrev"),
    (r"cogs", ["cost of revenue", "cost of goods sold", "cost of sales"], "abbrev"),
    (r"sg&a|sga", ["selling general and administrative expenses"], "abbrev"),
    (r"r&d", ["research and development expenses"], "abbrev"),
    (r"d&a", ["depreciation and amortization"], "abbrev"),
    (r"ebit", ["operating income"], "synonym"),
    (r"ebitda", ["operating income", "depreciation and amortization"], "synonym"),
    (r"fcf|free cash flow", ["net cash provided by operating activities", "capital expenditures"], "synonym"),
    (r"opex", ["operating expenses"], "abbrev"),
    (r"pp&e|ppe", ["property plant and equipment"], "abbrev"),
    (r"ar", ["accounts receivable"], "skip"),
    (r"capital leases?", ["finance lease", "finance leases", "finance lease liabilities"], "synonym"),
    (r"finance leases?", ["capital lease"], "synonym"),
    (r"sales|top line|top-line", ["revenue", "net sales"], "synonym"),
    (r"revenues?", ["net sales", "sales"], "synonym"),
    (r"profit|bottom line|bottom-line|earnings", ["net income"], "synonym"),
    (r"net income", ["net earnings", "net profit"], "synonym"),
    (r"debt maturit(?:y|ies)|maturing|due dates?|when (?:is|are) .* due", ["maturities of long-term debt", "principal payments", "thereafter"], "synonym"),
    (r"revolver|revolving (?:credit )?(?:facility|line)|credit (?:facility|line)", ["total commitments", "borrowing capacity", "available"], "synonym"),
    (r"grow(?:th|n|s)?|increase[ds]?|decline[ds]?|decrease[ds]?", ["increase", "decrease", "compared with"], "synonym"),
    (r"debt|borrowings?|loans?", ["long-term debt", "notes", "credit facility", "principal"], "synonym"),
    (r"liquidity", ["cash and cash equivalents", "credit facility", "capital resources"], "synonym"),
    (r"equity", ["stockholders equity", "shareholders equity"], "synonym"),
    (r"balance sheet", ["total assets", "total liabilities", "consolidated balance sheets"], "synonym"),
    (r"income statement|p&l|profit and loss", ["consolidated statements of income", "operating income", "net income"], "synonym"),
    (r"risks?", ["risk factors", "adverse", "could harm"], "synonym"),
    (r"supply chain", ["suppliers", "single-source", "component", "supply disruptions"], "synonym"),
    (r"going concern", ["substantial doubt"], "synonym"),
    (r"material weakness(?:es)?", ["internal control over financial reporting", "remediation"], "synonym"),
    (r"customer concentration|largest customer", ["accounted for", "percent of revenue", "concentration of credit risk"], "synonym"),
    (r"lawsuits?|litigation|legal", ["legal proceedings", "contingencies"], "synonym"),
    (r"employees|headcount", ["employed approximately", "human capital"], "synonym"),
    (r"buybacks?", ["share repurchases", "repurchase program"], "synonym"),
    (r"guidance|outlook", ["expects", "forecast", "outlook"], "synonym"),
]
_COMPILED = [(re.compile(rf"(?<![\w&]){pat}(?![\w&])", re.I), exp, kind) for pat, exp, kind in _ENTRIES if kind != "skip"]


def expand_terms(question: str) -> tuple[str, list[str], list[str]]:
    """Return (question with abbreviations spelled out, synonym words for BM25, human-readable notes)."""
    standalone = question
    synonyms: list[str] = []
    notes: list[str] = []
    for rx, exp, kind in _COMPILED:
        m = rx.search(question)
        if not m:
            continue
        if kind == "abbrev":
            full = exp[0]
            standalone = rx.sub(lambda mm: f"{mm.group(0)} ({full})", standalone, count=1)
            notes.append(f"{m.group(0)} → {full}")
        synonyms.extend(exp)
        if kind == "synonym":
            notes.append(f"{m.group(0)} ~ {', '.join(exp[:3])}")
    return standalone, list(dict.fromkeys(synonyms)), notes
