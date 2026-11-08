"""RagPipeline: ties ingestion, retrieval, rewriting, generation and attribution into one `ask()` call."""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from finrag.core.config import Settings
from finrag.core.schema import Answer, FilingMeta, RetrievedChunk
from finrag.generation import build_citations, generate_answer, make_llm, verify_answer
from finrag.generation.attribution import is_abstention
from finrag.generation.extractive import ABSTAIN
from finrag.ingestion import ingest_manifest
from finrag.ingestion.chunker import ChunkerConfig
from finrag.query import CompanyDirectory, QueryRewriter
from finrag.retrieval import Retriever, build_index, load_index, save_index

MODES = ("dense", "sparse", "hybrid")


class RagPipeline:
    def __init__(self, retriever: Retriever, filings: list[FilingMeta], llm=None, top_k: int = 6, today: date | None = None):
        self.retriever, self.filings, self.llm, self.top_k = retriever, filings, llm, top_k
        self.today = today
        self.directory = CompanyDirectory.from_filings(filings) if filings else CompanyDirectory.from_chunks(retriever.chunks)

    # ---------------------------------------------------------------- construction
    @classmethod
    def build(cls, filings: list[FilingMeta], settings: Settings | None = None, chunker: ChunkerConfig | None = None,
              save: bool = True, today: date | None = None) -> "RagPipeline":
        s = settings or Settings.from_env()
        chunks = ingest_manifest(filings, chunker)
        retriever = build_index(chunks, s.embedder, s.candidates, s.rrf_k)
        if save:
            save_index(retriever, s.index_dir)
            (s.index_dir / "filings.json").write_text(json.dumps([f.to_dict() for f in filings], indent=1))
        return cls(retriever, filings, make_llm(s.llm), s.top_k, today)

    @classmethod
    def load(cls, settings: Settings | None = None, today: date | None = None) -> "RagPipeline":
        s = settings or Settings.from_env()
        if not (s.index_dir / "meta.json").exists():
            raise FileNotFoundError(f"No index at {s.index_dir}. Run `finrag ingest` first (or `finrag demo`).")
        retriever = load_index(s.index_dir)
        fp = s.index_dir / "filings.json"
        filings = [FilingMeta.from_dict(d) for d in json.loads(fp.read_text())] if fp.exists() else []
        return cls(retriever, filings, make_llm(s.llm), s.top_k, today)

    # ---------------------------------------------------------------- querying
    def rewriter(self, mode: str) -> QueryRewriter:
        return QueryRewriter(self.directory, self.llm, self.today, mode)

    def ask(self, question: str, mode: str = "hybrid", rewrite: str = "rules", k: int | None = None,
            history: list[str] | None = None, use_llm: bool = True) -> Answer:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = k or self.top_k
        t = {}
        t0 = time.perf_counter()
        rq = self.rewriter(rewrite).rewrite(question, history)
        t["rewrite"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        retrieved = self.retriever.search(rq.standalone, k, mode, rq.filters, rq.sparse_query)
        t["retrieve"] = (time.perf_counter() - t0) * 1000

        warnings: list[str] = []
        if rq.clarification:
            warnings.append(rq.clarification)
        outside = [r for r in retrieved if r.outside_filter]
        if outside and rq.filters:
            warnings.append("Some retrieved passages come from filings outside the requested company/period (that filing may not be in the corpus).")

        t0 = time.perf_counter()
        llm = self.llm if use_llm else None
        if rq.unknown_entities and not rq.filters.get("ticker"):
            text, gen = (f"{ABSTAIN} “{rq.unknown_entities[0]}” is not in the corpus, so I will not answer from a different company's filings.", "guard")
        elif retrieved and all(r.outside_filter for r in retrieved) and any(rq.filters.get(x) for x in ("ticker", "fiscal_year", "fiscal_period", "filing_type")):
            text, gen = (f"{ABSTAIN} The corpus has no filing matching the requested company/period, so I will not answer from a different one.", "guard")
        else:
            try:
                text, gen = generate_answer(question, retrieved, llm, rq.standalone, rq.assumptions)
            except Exception as e:      # an LLM outage must not take the whole app down
                warnings.append(f"LLM call failed ({type(e).__name__}: {e}); fell back to extractive answer.")
                text, gen = generate_answer(question, retrieved, None, rq.standalone, rq.assumptions)
        t["generate"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        claims, stats = verify_answer(text, retrieved)
        t["verify"] = (time.perf_counter() - t0) * 1000
        abstained = is_abstention(text.split("\n")[-1]) or is_abstention(text)
        citations = build_citations(claims, retrieved)
        if not abstained and not citations:
            warnings.append("The answer carries no valid citations; treat it as unverified.")
        if stats["support_rate"] is not None and stats["support_rate"] < 1.0:
            warnings.append("Some sentences could not be matched to their cited source; check the claim details.")
        t["total"] = sum(t.values())
        return Answer(question=question, text=text, abstained=abstained, citations=citations, claims=claims,
                      support_rate=stats["support_rate"], citation_coverage=stats["citation_coverage"],
                      invalid_citations=stats["invalid_citations"], retrieved=retrieved, rewrite=rq, mode=mode,
                      generator=gen, timings_ms=t, warnings=warnings)

    def compare(self, question: str, k: int = 6, rewrite: str = "rules", history: list[str] | None = None) -> dict:
        """Run dense / sparse / hybrid retrieval on the same rewritten query (no generation)."""
        rq = self.rewriter(rewrite).rewrite(question, history)
        out = {"rewrite": rq.to_dict(), "results": {}}
        for m in MODES:
            t0 = time.perf_counter()
            res = self.retriever.search(rq.standalone, k, m, rq.filters, rq.sparse_query)
            out["results"][m] = {"ms": round((time.perf_counter() - t0) * 1000, 2), "chunks": [r.to_dict(text_chars=400) for r in res]}
        return out

    def info(self) -> dict:
        ch = self.retriever.chunks
        by_doc: dict[str, dict] = {}
        for c in ch:
            d = by_doc.setdefault(c.doc_id, {"doc_id": c.doc_id, "company": c.company, "ticker": c.ticker, "filing_type": c.filing_type,
                                             "fiscal_year": c.fiscal_year, "fiscal_period": c.fiscal_period, "period_end": c.period_end,
                                             "chunks": 0, "tables": 0})
            d["chunks"] += 1
            d["tables"] += c.element_type in ("table", "table_rows")
        return {"n_chunks": len(ch), "n_filings": len(by_doc), "embedder": self.retriever.dense.embedder.name,
                "llm": self.llm.name if self.llm else "none (extractive)", "filings": sorted(by_doc.values(), key=lambda d: (d["ticker"], d["period_end"]))}
