"""Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum_r  w_r / (k + rank_r(d)).

Rank-based, so it needs no score normalisation between BM25 (unbounded) and cosine (0..1).
"""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int = 60, weights: dict[str, float] | None = None
                           ) -> list[tuple[int, float, dict[str, int]]]:
    """rankings: {retriever_name: [doc_id best-first]} -> [(doc_id, fused_score, {retriever: rank})]."""
    weights = weights or {}
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, docs in rankings.items():
        w = weights.get(name, 1.0)
        for r, doc in enumerate(docs, start=1):
            fused[doc] = fused.get(doc, 0.0) + w / (k + r)
            ranks.setdefault(doc, {})[name] = r
    return [(d, s, ranks[d]) for d, s in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
