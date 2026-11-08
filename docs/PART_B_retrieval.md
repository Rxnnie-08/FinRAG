# Part B: hybrid retrieval

Owns: `retrieval/` (number-preserving tokenizer, in-house BM25, dense embedders, RRF, metadata filters, index persistence).

Contract: `build_index(chunks, embedder) -> Retriever`; `Retriever.search(query, k, mode, filters, sparse_query) -> list[RetrievedChunk]`
with `mode` in `dense | sparse | hybrid`. Filters apply before ranking; if too few chunks match they are relaxed and the extras
are flagged `outside_filter`.

Test: `pytest tests/b_retrieval` (needs Part A merged). Embedders: `lsa` (offline), `st:<model>`, `ollama:<model>`.
Your experiment: swap in `st:BAAI/bge-small-en-v1.5` and re-run `finrag eval`; report dense vs sparse vs hybrid honestly.
