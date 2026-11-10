"""Qdrant: the project's vector database. It runs ALL THREE retrieval modes.

One collection, one point per chunk:
  dense vector   our embedding (LSA / sentence-transformers / Ollama), cosine distance
  sparse vector  BM25 term weights. We store tf*(k1+1)/(tf + k1*(1-b+b*len/avglen)) per term and let Qdrant apply the
                 IDF modifier at query time, which makes the sparse score exactly Okapi BM25 (the test suite checks it
                 against the in-house BM25 in bm25.py). Terms are our number-preserving tokens, hashed to u32 ids.
  payload        chunk metadata (ticker, form, fiscal year/period, item, ...)

Modes:  dense  -> query the dense vector;  sparse -> query the sparse vector;
        hybrid -> ONE Query API call: prefetch dense and sparse candidates, fuse with reciprocal rank fusion (k=60) inside Qdrant.
Metadata filters are computed in `hybrid.apply_filters` (so filter relaxation is identical for every store) and sent to
Qdrant as an id filter on every prefetch.

Known behaviours: Qdrant counts RRF ranks from 0 (scores sit a constant above the textbook formula; the order is the same), and
chunks with exactly equal scores (e.g. identical boilerplate in two filings) are fused in arbitrary order inside Qdrant. Dense-only and
sparse-only results are made deterministic here (ties broken by chunk position); hybrid results may reorder exact ties between runs.

The in-house BM25 (bm25.py) stays as the reference implementation and as the engine for the `numpy` (no-database) store.

Deployment modes (pick one):
  local   default. Qdrant runs inside this process and stores to <index>/qdrant. No install, but ONE process may open the
          folder at a time (so `finrag ask` fails while `finrag serve` is running). Qdrant recommends local mode for small data only.
  server  set FINRAG_QDRANT_URL=http://localhost:6333  (docker run -p 6333:6333 qdrant/qdrant). Many processes, large data.
          Optional QDRANT_API_KEY and FINRAG_QDRANT_COLLECTION (default "finrag_chunks").
"""
from __future__ import annotations

import os
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from finrag.core.schema import Chunk

from .bm25 import BM25Index
from .dense import DenseIndex, Embedder
from .tokenizer import tokenize_query

BATCH = 256
DENSE, SPARSE = "dense", "sparse"


def server_url() -> str | None:
    return os.environ.get("FINRAG_QDRANT_URL") or None


def collection_name() -> str:
    return os.environ.get("FINRAG_QDRANT_COLLECTION", "finrag_chunks")


def term_id(term: str) -> int:
    """Stable u32 id for a token (no vocabulary file to keep in sync; collisions among ~10^5 terms are negligible)."""
    return zlib.crc32(term.encode("utf-8"))


def payload_of(c: Chunk) -> dict:
    return {"ticker": c.ticker, "filing_type": c.filing_type, "fiscal_year": c.fiscal_year, "fiscal_period": c.fiscal_period,
            "period_end": c.period_end, "item": c.item, "element_type": c.element_type, "chunk_id": c.chunk_id}


def sparse_docs(bm25: BM25Index) -> list[dict[int, float]]:
    """Per-document BM25 term weights (everything except IDF, which Qdrant multiplies in)."""
    docs: list[dict[int, float]] = [defaultdict(float) for _ in range(bm25.n_docs)]
    for term, plist in bm25.postings.items():
        tid = term_id(term)
        for doc, tf in plist:
            dl = bm25.doc_len[doc]
            docs[doc][tid] += tf * (bm25.k1 + 1) / (tf + bm25.k1 * (1 - bm25.b + bm25.b * dl / (bm25.avgdl or 1)))
    return docs


def _stable(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Qdrant orders equal scores arbitrarily; break ties by chunk position so results are reproducible across runs and stores."""
    return sorted(pairs, key=lambda x: (-round(x[1], 5), x[0]))


def _open(path: str | Path | None) -> QdrantClient:
    url = server_url()
    try:
        if url:
            return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
        if path is None:
            return QdrantClient(":memory:")
        Path(path).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    except RuntimeError as e:
        if "already accessed" in str(e):
            raise RuntimeError("The Qdrant index folder is open in another finrag process (for example `finrag serve`). Stop that "
                               "process, or run Qdrant as a server and set FINRAG_QDRANT_URL=http://localhost:6333.") from e
        raise


class QdrantIndex(DenseIndex):
    """Dense + sparse + hybrid search in Qdrant. Subclasses DenseIndex so `.search()` is the dense query, as elsewhere."""
    store_name = "qdrant"
    native_hybrid = True

    def __init__(self, embedder: Embedder, path: str | Path | None = None):
        super().__init__(embedder)
        self.path = Path(path) if path else None
        self.server = bool(server_url())
        self.collection = collection_name()
        self.client = _open(path)
        self.n = 0

    # ------------------------------------------------------------------ building
    def _create(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)})

    def fit(self, texts: list[str], payloads: list[dict] | None = None, bm25: BM25Index | None = None) -> "QdrantIndex":
        self.embedder.fit(texts)
        sparse = sparse_docs(bm25 or BM25Index().fit(texts))
        created = False
        for i in range(0, len(texts), BATCH):
            vecs = np.asarray(self.embedder.encode(texts[i:i + BATCH]), dtype="float32")
            if not created:
                self._create(vecs.shape[1])
                created = True
            self.client.upsert(self.collection, points=[
                models.PointStruct(id=i + j, vector={DENSE: v.tolist(), SPARSE: models.SparseVector(indices=list(sparse[i + j].keys()), values=list(sparse[i + j].values()))},
                                   payload=(payloads[i + j] if payloads else {"idx": i + j}))
                for j, v in enumerate(vecs)])
        self.n = len(texts)
        return self

    @classmethod
    def attach(cls, embedder: Embedder, path: str | Path | None) -> "QdrantIndex":
        """Open an index an earlier run already wrote (no re-embedding)."""
        idx = cls(embedder, path)
        if not idx.client.collection_exists(idx.collection):
            raise FileNotFoundError(f"Qdrant collection '{idx.collection}' not found. Rebuild the index (finrag demo / finrag ingest).")
        if not idx.client.get_collection(idx.collection).config.params.sparse_vectors:
            raise FileNotFoundError("This index was built by an older version (no sparse vectors). Rebuild it: finrag demo / finrag ingest.")
        idx.n = idx.client.count(idx.collection).count
        return idx

    def export(self, path: str | Path) -> "QdrantIndex":
        """Copy this index into a persistent local folder (used when saving an in-memory index)."""
        dst = QdrantIndex(self.embedder, path)
        offset, first = None, True
        while True:
            pts, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_vectors=True, with_payload=True)
            if pts and first:
                dst._create(len(pts[0].vector[DENSE]))
                first = False
            if pts:
                dst.client.upsert(dst.collection, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in pts])
            if offset is None:
                break
        dst.n = self.n
        return dst

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ searching
    def _limit(self, k: int, allowed: set[int] | None) -> int:
        return 0 if self.n == 0 else min(k, len(allowed) if allowed is not None else self.n)

    @staticmethod
    def _filter(allowed: set[int] | None):
        return models.Filter(must=[models.HasIdCondition(has_id=sorted(allowed))]) if allowed is not None else None

    def _dense_vec(self, query: str) -> list[float]:
        return np.asarray(self.embedder.encode([query], is_query=True)[0], dtype="float32").tolist()

    @staticmethod
    def _sparse_vec(query: str) -> models.SparseVector | None:
        ids = list(dict.fromkeys(term_id(t) for t in tokenize_query(query)))      # unique terms, weight 1 (as in bm25.py)
        return models.SparseVector(indices=ids, values=[1.0] * len(ids)) if ids else None

    def search(self, query: str, k: int = 50, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """Dense (cosine) search."""
        limit = self._limit(k, allowed)
        if limit <= 0:
            return []
        res = self.client.query_points(self.collection, query=self._dense_vec(query), using=DENSE, limit=limit,
                                       query_filter=self._filter(allowed)).points
        return _stable([(int(p.id), float(p.score)) for p in res])

    def search_sparse(self, query: str, k: int = 50, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """BM25 search over the sparse vectors (IDF applied by Qdrant)."""
        limit, sv = self._limit(k, allowed), self._sparse_vec(query)
        if limit <= 0 or sv is None:
            return []
        res = self.client.query_points(self.collection, query=sv, using=SPARSE, limit=limit, query_filter=self._filter(allowed)).points
        return _stable([(int(p.id), float(p.score)) for p in res if p.score > 0])

    def search_hybrid(self, dense_q: str, sparse_q: str, k: int = 50, allowed: set[int] | None = None, rrf_k: int = 60,
                      weights: tuple[float, float] | None = None, candidates: int = 50) -> list[tuple[int, float]]:
        """Hybrid search in ONE Qdrant call: dense and sparse prefetch, reciprocal rank fusion inside the database."""
        limit, sv = self._limit(k, allowed), self._sparse_vec(sparse_q)
        if limit <= 0:
            return []
        if sv is None:                                   # nothing to match by keyword: hybrid degrades to dense
            return self.search(dense_q, k, allowed)
        flt, n = self._filter(allowed), max(candidates, k)
        rrf = models.Rrf(k=rrf_k, weights=list(weights)) if weights and tuple(weights) != (1.0, 1.0) else models.Rrf(k=rrf_k)
        res = self.client.query_points(
            self.collection,
            prefetch=[models.Prefetch(query=self._dense_vec(dense_q), using=DENSE, limit=n, filter=flt),
                      models.Prefetch(query=sv, using=SPARSE, limit=n, filter=flt)],
            query=models.RrfQuery(rrf=rrf), limit=limit).points
        return _stable([(int(p.id), float(p.score)) for p in res])


QdrantDenseIndex = QdrantIndex          # earlier name
