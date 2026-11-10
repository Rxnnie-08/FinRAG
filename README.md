# FinRAG: Financial Document Intelligence

A RAG question-answering system over SEC filings (10-K, 10-Q, 8-K) that answers with the source row beside it.
Course mini project: *Prompt Engineering and Optimization for Generative AI*.

```
EDGAR HTML/PDF ─► structure-aware parse ─► chunks (tables atomic, header repeated, footnotes attached, XBRL facts)
                                              │ embed (LSA / sentence-transformers / Ollama)
                                              ▼
                        Qdrant: dense vector + sparse BM25 vector + chunk metadata, one collection
                                              │
question ─► rewriter ─► metadata filter ─► Qdrant Query API: dense prefetch + sparse prefetch ─► RRF fusion (inside Qdrant)
(company, fiscal period, glossary)                                              │
                                                                                ▼
                        DSPy answer program on Ollama (prompt optimised on the gold set) ─► claim check ─► UI
```

Technologies: Python, BeautifulSoup/lxml, **Qdrant** (vector database), Qdrant-native hybrid search (dense + sparse BM25 vectors, server-side RRF), **DSPy** (prompt program and optimiser),
**Ollama** (local LLM), Flask, plain JavaScript UI, pytest.

## Run it

Prerequisites: Python 3.10+, and [Ollama](https://ollama.com) with a model pulled (`ollama pull llama3.1:8b`). Without Ollama running, the app still works
and answers by extractive lookup (the UI shows a note saying so).

```bash
pip install -e ".[dev]"      # includes Qdrant and DSPy
finrag demo                  # builds a synthetic corpus + Qdrant index in ./finrag_data
finrag ask "What were Northwind Robotics' total assets at the end of fiscal 2024?" -v
finrag optimize              # DSPy: tune the answer prompt on the gold questions (needs Ollama); saved and loaded automatically
finrag serve                 # web UI at http://127.0.0.1:8000
finrag eval                  # benchmark: 3 retrievers x rewriting on/off, with bootstrap CIs
pytest
```

Model: `FINRAG_LLM=ollama:<model>` (default `ollama:llama3.1:8b`; see `ollama list`), or `anthropic:<model>`, `openai:<model>@<base_url>`, or `none`.
Qdrant: by default it runs inside the process and stores to `<home>/index/qdrant`, which only ONE process can open at a time (so `finrag ask` fails while
`finrag serve` runs). For several processes or bigger corpora run a server, `docker run -p 6333:6333 qdrant/qdrant`, and set `FINRAG_QDRANT_URL=http://localhost:6333`.
`finrag demo --vector-store numpy` skips the database (in-memory matrix) for comparison.

Your own filing: start `finrag serve`, then use **Add a filing** in the left panel to upload a 10-K/10-Q/8-K (.htm or .pdf). Company, form and period are read from the file; fix them under "Details" if needed. Name the company in your question so the right filing is searched.

Real filings in bulk: `export SEC_USER_AGENT="Your Name you@uni.edu"`, then `finrag fetch --tickers AAPL MSFT --limit 2 --out edgar_data`
and `finrag ingest --manifest edgar_data/manifest.json`.
FinanceBench: `pip install -e ".[pdf]"`, `finrag financebench --dest financebench_data`, then `finrag ingest` and `finrag eval --gold ...`.
Better embeddings: `ollama pull nomic-embed-text` and `FINRAG_EMBEDDER=ollama:nomic-embed-text` (or `pip install -e ".[st]"` and `st:BAAI/bge-small-en-v1.5`).

## Team split (one folder of the repo each; see `docs/PART_*.md`)

| Part | Owner area | Paths |
|---|---|---|
| A | Data and ingestion; shared contracts | `src/finrag/core`, `src/finrag/ingestion`, `sample_data`, `pyproject.toml`, `tests/a_core_ingestion`, `tests/conftest.py` |
| B | Hybrid retrieval | `src/finrag/retrieval`, `tests/b_retrieval` |
| C | Query rewriting, generation, attribution | `src/finrag/query`, `src/finrag/generation`, `tests/c_query_generation` |
| D | Evaluation, pipeline, CLI, server, UI | `src/finrag/pipeline.py`, `cli.py`, `evaluation`, `server`, `tests/d_eval_app` |

Everything imports the dataclasses in `finrag/core/schema.py`. Change a field there only after telling the team.
Merge order: A first (B, C, D import `finrag.core`), then B and C in any order, then D.

## Results on real filings

_To be filled in from `docs/REAL_DATA_RUNBOOK.md`: FinanceBench table, EDGAR HTML table, judge-vs-human agreement._

## Smoke-test results on the synthetic corpus (38 questions, k=5, heuristic judge)

| Setup | Recall@3 | Answer correct | Refuses when it should |
|---|---|---|---|
| dense | 74% | 68% | 50% |
| sparse (BM25) | 94% | 68% | 50% |
| hybrid | 91% | 65% | 50% |
| dense + rewrite | 97% | 94% | 100% |
| sparse + rewrite | 100% | 91% | 100% |
| hybrid + rewrite | 100% | 97% | 100% |

Read these with care.
- **[Certain]** This is a smoke test on a corpus we generated, with two fictional companies and 38 questions. It is not evidence for the report. Rerun on real filings and FinanceBench.
- **[Certain]** Rewriting gives the large, statistically clear gain. Hybrid beats dense on recall@3 (paired bootstrap CI excludes 0).
- **[Certain]** Hybrid is not better than BM25 here (BM25 has higher MRR). Number-heavy questions favour keyword matching, and the dense model is a weak offline LSA baseline. Do not claim hybrid wins until a neural embedder has been tried on real data.
- **[Likely]** The hybrid-over-dense gap is inflated by the weak LSA embedder.

## Known limitations

- sec.gov was unreachable where this was built: `EdgarClient` is tested with a mocked session only, and the FinanceBench PDF path is untested end to end.
- The DSPy optimiser has only been run against a stub model (no LLM where this was built). The machinery is tested; whether it improves answers needs a real Ollama run on real questions, and it may not. Only the answer prompt is optimised; retrieval, rewriting and the claim checker are not learned. With no LLM, DSPy is idle and answers are extractive.
- Dense, sparse (BM25 weights stored as a Qdrant sparse vector with IDF modifier) and hybrid (RRF) all run inside Qdrant. Our `bm25.py` is kept as the reference (tests check Qdrant's sparse scores against it) and powers the optional `numpy` store. Caveat: Qdrant orders exact score ties arbitrarily, and the corpus has many identical boilerplate chunks, so a few rankings differ slightly from the NumPy store (hybrid+rewrite answer-correct 91% vs 97% on 38 questions, from tie order in 2 questions). Old index folders must be rebuilt.
- The default answerer is extractive (a lookup baseline). LLM backends are implemented but untested without keys.
- The claim check (numeric match + word overlap) is a heuristic and reports a rate; it does not guarantee faithfulness.
- Page numbers from EDGAR HTML are approximate (page-break markers); PDF pages are exact.
- The UI has no authentication and binds to localhost. Do not expose it to a network.
- The pickled LSA index must only be loaded from folders you built yourself.
