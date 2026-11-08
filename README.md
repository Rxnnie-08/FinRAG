# FinRAG: Financial Document Intelligence

A RAG question-answering system over SEC filings (10-K, 10-Q, 8-K) that answers with the source row beside it.
Course mini project: *Prompt Engineering and Optimization for Generative AI*.

```
EDGAR HTML/PDF ─► structure-aware parse ─► chunks (tables atomic, header repeated, footnotes attached, XBRL facts)
                                              │
question ─► rewriter (company, fiscal period, glossary) ─► dense + BM25 ─► RRF fusion ─► answer ─► claim check ─► UI
```

## Run it (no API key, no internet)

```bash
pip install -e ".[dev]"
finrag demo                  # builds a synthetic corpus + index in ./finrag_data
finrag ask "What were Northwind Robotics' total assets at the end of fiscal 2024?" -v
finrag serve --port 8000     # web UI at http://127.0.0.1:8000
finrag eval                  # benchmark: 3 retrievers x rewriting on/off, with bootstrap CIs
pytest
```

Real filings: `export SEC_USER_AGENT="Your Name you@uni.edu"`, then `finrag fetch --tickers AAPL MSFT --limit 2 --out edgar_data`
and `finrag ingest --manifest edgar_data/manifest.json`.
FinanceBench: `pip install -e ".[pdf]"`, `finrag financebench --dest financebench_data`, then `finrag ingest` and `finrag eval --gold ...`.
Better embeddings: `pip install -e ".[st]"` and `FINRAG_EMBEDDER=st:BAAI/bge-small-en-v1.5`. LLM answers: `FINRAG_LLM=anthropic:<model>`.

## Team split (one folder of the repo each)

| Part | Owner area | Paths |
|---|---|---|
| A | Data and ingestion; shared contracts | `src/finrag/core`, `src/finrag/ingestion`, `sample_data`, `pyproject.toml`, `tests/a_core_ingestion`, `tests/conftest.py` |
| B | Hybrid retrieval | `src/finrag/retrieval`, `tests/b_retrieval` |
| C | Query rewriting, generation, attribution | `src/finrag/query`, `src/finrag/generation`, `tests/c_query_generation` |
| D | Evaluation, pipeline, CLI, server, UI | `src/finrag/pipeline.py`, `cli.py`, `evaluation`, `server`, `tests/d_eval_app` |

Everything imports the dataclasses in `finrag/core/schema.py`. Change a field there only after telling the team.


## Results on the synthetic corpus (38 questions, k=5, heuristic judge)

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

