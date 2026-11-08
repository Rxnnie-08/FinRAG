"""Retriever: dense, sparse or hybrid (RRF), with metadata filters.

Filters are applied *before* ranking (pre-filter) so a question about FY2024 cannot be answered from FY2023
text. If the filtered pool has fewer than k results (e.g. the requested filing is not in the corpus) the pool is
relaxed and those results are flagged `outside_filter=True`, so the pipeline can warn instead of staying silent.
"""
from __future__ import annotations

from typing import Any

from finrag.core.schema import Chunk, RetrievedChunk

from .bm25 import BM25Index
from .dense import DenseIndex
from .fusion import reciprocal_rank_fusion

MODES = ("dense", "sparse", "hybrid")


def _as_set(v: Any) -> set | None:
    if v is None or v == "" or v == []:
        return None
    if isinstance(v, (list, tuple, set)):
        return {x for x in v}
    return {v}


def apply_filters(chunks: list[Chunk], filters: dict[str, Any] | None) -> set[int] | None:
    """Return allowed chunk indices (None = no filtering). Keys: ticker, filing_type, fiscal_year, fiscal_period,
    latest (bool: keep only the most recent period_end per ticker among the matches)."""
    if not filters:
        return None
    tick = {str(t).upper() for t in (_as_set(filters.get("ticker")) or [])} or None
    ftype = _as_set(filters.get("filing_type"))
    fy = {int(y) for y in (_as_set(filters.get("fiscal_year")) or [])} or None
    fper = _as_set(filters.get("fiscal_period"))
    allowed = set()
    for i, c in enumerate(chunks):
        if tick and c.ticker.upper() not in tick:
            continue
        if ftype and c.filing_type not in ftype:
            continue
        if fy and c.fiscal_year not in fy:
            continue
        if fper and c.fiscal_period not in fper:
            continue
        allowed.add(i)
    if filters.get("latest") and allowed:
        best: dict[str, str] = {}
        for i in allowed:
            c = chunks[i]
            best[c.ticker] = max(best.get(c.ticker, ""), c.period_end)
        allowed = {i for i in allowed if chunks[i].period_end == best[chunks[i].ticker]}
    return allowed if any(filters.get(k) for k in ("ticker", "filing_type", "fiscal_year", "fiscal_period", "latest")) else None


class Retriever:
    def __init__(self, chunks: list[Chunk], dense: DenseIndex, sparse: BM25Index, candidates: int = 50, rrf_k: int = 60,
                 weights: dict[str, float] | None = None):
        self.chunks, self.dense, self.sparse = chunks, dense, sparse
        self.candidates, self.rrf_k = candidates, rrf_k
        self.weights = weights or {"dense": 1.0, "sparse": 1.0}

    def _rank(self, mode: str, dense_q: str, sparse_q: str, allowed: set[int] | None, n: int):
        out: dict[str, list[tuple[int, float]]] = {}
        if mode in ("dense", "hybrid"):
            out["dense"] = self.dense.search(dense_q, n, allowed)
        if mode in ("sparse", "hybrid"):
            out["sparse"] = self.sparse.search(sparse_q, n, allowed)
        return out

    def search(self, query: str, k: int = 6, mode: str = "hybrid", filters: dict[str, Any] | None = None,
               sparse_query: str | None = None) -> list[RetrievedChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        sparse_query = sparse_query or query
        allowed = apply_filters(self.chunks, filters)
        results = self._fuse(mode, query, sparse_query, allowed, k)
        if allowed is not None and len(results) < k:
            seen = {r.chunk.chunk_id for r in results}
            extra = [r for r in self._fuse(mode, query, sparse_query, None, k + len(results)) if r.chunk.chunk_id not in seen]
            for r in extra[: k - len(results)]:
                r.outside_filter = True
                results.append(r)
            for n, r in enumerate(results, start=1):
                r.rank = n
        return results

    def _fuse(self, mode: str, dense_q: str, sparse_q: str, allowed, k: int) -> list[RetrievedChunk]:
        ranked = self._rank(mode, dense_q, sparse_q, allowed, max(self.candidates, k))
        scores = {name: dict(pairs) for name, pairs in ranked.items()}
        if mode == "hybrid":
            fused = reciprocal_rank_fusion({n: [d for d, _ in p] for n, p in ranked.items()}, self.rrf_k, self.weights)
            rows = [(d, s, r) for d, s, r in fused[:k]]
        else:
            name = mode
            rows = [(d, s, {name: i}) for i, (d, s) in enumerate(ranked[name][:k], start=1)]
        out = []
        for n, (d, s, r) in enumerate(rows, start=1):
            out.append(RetrievedChunk(chunk=self.chunks[d], score=float(s), rank=n, method=mode, ranks=r,
                                      scores={name: scores[name][d] for name in scores if d in scores[name]}))
        return out
