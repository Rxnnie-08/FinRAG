# Part B: hybrid retrieval

Owns: `retrieval/` (number-preserving tokenizer, in-house BM25, dense embedders, RRF, metadata filters, index persistence).

Contract: `build_index(chunks, embedder) -> Retriever`; `Retriever.search(query, k, mode, filters, sparse_query) -> list[RetrievedChunk]`
with `mode` in `dense | sparse | hybrid`. Filters apply before ranking; if too few chunks match they are relaxed and the extras
are flagged `outside_filter`.

Test: `pytest tests/b_retrieval` (needs Part A merged). Embedders: `lsa` (offline), `st:<model>`, `ollama:<model>`.
Your experiment: swap in `st:BAAI/bge-small-en-v1.5` and re-run `finrag eval`; report dense vs sparse vs hybrid honestly.

Vector database: `retrieval/vectorstore.py` (`QdrantIndex`, the default; stores a dense and a sparse BM25 vector per chunk) behind the same `search(query, k, allowed)` interface as the NumPy index. Each chunk is a Qdrant point (id = chunk position, payload = chunk metadata). Metadata filters still run in `hybrid.apply_filters`, and the allowed positions go to Qdrant as an id filter, so filter-relaxation behaves identically across stores. Dense, sparse and hybrid search all run in Qdrant (hybrid = two prefetches fused by `RrfQuery`); `bm25.py` and `reciprocal_rank_fusion` remain as the reference/NumPy-store path. Local mode = one process per index folder; `FINRAG_QDRANT_URL` selects a server. Test: `pytest tests/b_retrieval/test_vectorstore.py`.

Embedders (`retrieval/dense.py`): default `ollama:nomic-embed-text` (with document/query prefixes), `st:<model>`, and `lsa` (offline baseline, also the automatic fallback with a UI note when Ollama is down). `resolve_embedder` never raises for a missing Ollama; `load_index` raises a clear error if an index built with an Ollama embedder is opened while Ollama is down.
