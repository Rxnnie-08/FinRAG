"""Shared data contracts. Every module in the system speaks in these types.

If you change a field here, tell the whole team: ingestion produces Chunks, retrieval indexes them,
generation cites them and evaluation scores against them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class FilingMeta:
    company: str
    ticker: str
    filing_type: str            # "10-K" | "10-Q" | "8-K"
    period_end: str             # ISO date of the period the filing covers
    accession: str = ""
    cik: str = ""
    fiscal_year: int | None = None
    fiscal_period: str = ""     # "FY" | "Q1".."Q3" ("" for 8-K)
    fiscal_year_end: str = "1231"   # MMDD
    filing_date: str = ""
    path: str = ""              # local HTML/PDF path
    url: str = ""               # canonical EDGAR URL, if known

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FilingMeta":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}-{self.filing_type}-{self.period_end}"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    company: str
    ticker: str
    filing_type: str
    fiscal_year: int | None
    fiscal_period: str
    period_end: str
    accession: str = ""
    source: str = ""
    item: str = ""              # "Item 7"
    section: str = ""           # "Management's Discussion and Analysis"
    note: str = ""              # "Note 8 - Debt"
    element_type: str = "text"  # text | table | table_rows | xbrl_facts
    table_id: str = ""
    caption: str = ""
    units: str = ""
    footnotes: list[str] = field(default_factory=list)
    table_data: dict[str, Any] | None = None   # {"header": [...], "rows": [[...]]} for UI + row-level citation
    page: int | None = None     # approximate (EDGAR HTML) or exact (PDF)
    block_start: int = 0
    block_end: int = 0
    url: str = ""

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}-{self.filing_type}-{self.period_end}"

    @property
    def context_header(self) -> str:
        """Short description prepended at index time (contextual retrieval)."""
        period = f"fiscal {self.fiscal_year}" if self.fiscal_year else ""
        if self.fiscal_period and self.fiscal_period != "FY":
            period += f" {self.fiscal_period}"
        parts = [self.company, self.ticker, self.filing_type, period, self.item, self.section, self.note, self.caption]
        return " | ".join(p for p in parts if p)

    @property
    def index_text(self) -> str:
        return f"{self.context_header}\n{self.text}"

    @property
    def label(self) -> str:
        fy = f"FY{self.fiscal_year}" if self.fiscal_year else self.period_end
        if self.fiscal_period and self.fiscal_period not in ("", "FY"):
            fy += f" {self.fiscal_period}"
        bits = [f"{self.ticker} {self.filing_type} {fy}", self.item, self.note or self.section, self.table_id]
        return " · ".join(b for b in bits if b)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    method: str = "hybrid"                       # dense | sparse | hybrid
    ranks: dict[str, int] = field(default_factory=dict)    # per-retriever rank (explainability)
    scores: dict[str, float] = field(default_factory=dict)
    outside_filter: bool = False                 # True if added after filters were relaxed

    def to_dict(self, text_chars: int = 1500) -> dict[str, Any]:
        c = self.chunk
        return {
            "rank": self.rank,
            "score": round(self.score, 5),
            "method": self.method,
            "ranks": self.ranks,
            "outside_filter": self.outside_filter,
            "chunk_id": c.chunk_id,
            "label": c.label,
            "element_type": c.element_type,
            "text": c.text if len(c.text) <= text_chars else c.text[:text_chars] + " …",
            "table_data": c.table_data,
            "footnotes": c.footnotes,
            "units": c.units,
            "meta": {
                "company": c.company, "ticker": c.ticker, "filing_type": c.filing_type,
                "fiscal_year": c.fiscal_year, "fiscal_period": c.fiscal_period, "period_end": c.period_end,
                "item": c.item, "section": c.section, "note": c.note, "table_id": c.table_id,
                "page": c.page, "accession": c.accession, "url": c.url,
            },
        }


@dataclass
class RewrittenQuery:
    original: str
    standalone: str                              # what the dense retriever and the generator see
    sparse_query: str                            # standalone + glossary expansions (BM25 only)
    filters: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    clarification: str = ""
    expansions: list[str] = field(default_factory=list)
    mode: str = "rules"                          # off | rules | llm
    used_llm: bool = False
    unknown_entities: list[str] = field(default_factory=list)   # company-like names that are not in the corpus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    n: int
    chunk_id: str
    label: str
    ticker: str
    filing_type: str
    fiscal_year: int | None
    fiscal_period: str
    accession: str
    item: str
    section: str
    note: str
    table_id: str
    page: int | None
    url: str
    span: str = ""                               # best supporting sentence / table row
    element_type: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimCheck:
    sentence: str
    citations: list[int]
    cited: bool
    supported: bool | None                       # None = not a factual claim / abstention
    numeric_ok: bool | None
    overlap: float
    span: str = ""
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool = False
    citations: list[Citation] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)
    support_rate: float | None = None
    citation_coverage: float | None = None
    invalid_citations: int = 0
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    rewrite: RewrittenQuery | None = None
    mode: str = "hybrid"
    generator: str = "extractive"
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "citations": [c.to_dict() for c in self.citations],
            "claims": [c.to_dict() for c in self.claims],
            "support_rate": self.support_rate,
            "citation_coverage": self.citation_coverage,
            "invalid_citations": self.invalid_citations,
            "retrieved": [r.to_dict() for r in self.retrieved],
            "rewrite": self.rewrite.to_dict() if self.rewrite else None,
            "mode": self.mode,
            "generator": self.generator,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "warnings": self.warnings,
        }
