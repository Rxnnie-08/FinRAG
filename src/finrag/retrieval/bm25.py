"""BM25 (Okapi) over an inverted index. Small, dependency-free, deterministic."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .tokenizer import tokenize, tokenize_query


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.n_docs = 0

    def fit(self, texts: list[str]) -> "BM25Index":
        self.postings = defaultdict(list)
        self.doc_len = []
        for i, text in enumerate(texts):
            toks = tokenize(text)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.n_docs = len(texts)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        return self

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        q = tokenize_query(query)
        scores: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(q):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for doc, tf in plist:
                if allowed is not None and doc not in allowed:
                    continue
                dl = self.doc_len[doc]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[doc] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
