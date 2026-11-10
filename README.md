# FinRAG: Financial Document Intelligence

A question-answering system over SEC filings (10-K, 10-Q, 8-K). It answers in plain language and shows the exact filing passage, table row and page behind every statement.

Course mini project: *Prompt Engineering and Optimization for Generative AI*.

## What it does

- **Reads filings the way an analyst does.** Tables are kept whole (or split by rows with the header repeated), footnotes stay attached to their rows, and XBRL facts are captured.
- **Understands the question.** A query rewriter resolves fiscal periods per company (a "Q2 of fiscal 2025" depends on the company's year-end), expands finance terms ("material weakness" ~ "internal control over financial reporting"), carries context over from the previous question, and refuses to guess when a company is unknown.
- **Filters before it ranks.** Company, form type, fiscal year and period are applied as metadata filters, so a 2023 number is never returned for a 2024 question.
- **Retrieves three ways inside one database.** Qdrant runs dense (meaning), sparse (BM25 keyword) and hybrid (both, fused with reciprocal rank fusion) search.
- **Writes answers with a DSPy program on a local LLM (Ollama)**, and checks every statement against its cited source.
- **Measures itself.** Recall@k, MRR, answer correctness, abstention and bootstrap confidence intervals, with paired comparisons between setups.

```
EDGAR HTML/PDF ─► structure-aware parse ─► chunks (tables, footnotes, XBRL facts, context header)
                                              │ embed (LSA / sentence-transformers / Ollama)
                                              ▼
                        Qdrant: dense vector + sparse BM25 vector + chunk metadata (one collection)
                                              │
question ─► rewriter ─► metadata filter ─► Qdrant Query API: dense prefetch + sparse prefetch ─► RRF fusion (inside Qdrant)
(company, fiscal period, glossary)                                              │
                                                                                ▼
                        DSPy answer program on Ollama (prompt optimised on the gold set) ─► claim check ─► UI
```

Technologies: Python, BeautifulSoup/lxml, **Qdrant** (vector database with native hybrid search), **DSPy** (prompt program and optimiser),
**Ollama** (local LLM), Flask, plain JavaScript UI, pytest.

## Run it

Prerequisites: Python 3.10+, and [Ollama](https://ollama.com) with a model pulled (`ollama pull llama3.1:8b`). Without Ollama running, the app still works
and answers by extractive lookup (the UI shows a note saying so).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]" --default-timeout=300 --retries 10   # includes Qdrant and DSPy
finrag demo                  # builds the sample corpus and the Qdrant index in ./finrag_data
finrag ask "What were Northwind Robotics' total assets at the end of fiscal 2024?" -v
finrag optimize              # DSPy: tune the answer prompt on the gold questions (needs Ollama); saved and loaded automatically
finrag serve                 # web UI at http://127.0.0.1:8000
finrag eval                  # benchmark: 3 retrievers x rewriting on/off, with bootstrap CIs
pytest                       # 91 tests
```

On macOS/Linux, activate with `source .venv/bin/activate`.

**Configuration (environment variables)**

| Variable | Meaning | Default |
|---|---|---|
| `FINRAG_LLM` | `ollama:<model>`, `anthropic:<model>`, `openai:<model>@<base_url>`, or `none` | `ollama:llama3.1:8b` |
| `FINRAG_EMBEDDER` | `lsa`, `ollama:<model>`, or `st:<model>` | `lsa` |
| `FINRAG_VECTOR_STORE` | `qdrant` or `numpy` (no database, for comparison) | `qdrant` |
| `FINRAG_QDRANT_URL` | address of a Qdrant server; unset = runs inside the process | unset |

**Qdrant modes.** By default Qdrant runs inside the process and stores to `<home>/index/qdrant`, which only ONE process can open at a time (so `finrag ask` fails while
`finrag serve` runs). For several processes or bigger corpora run a server, `docker run -p 6333:6333 qdrant/qdrant`, and set `FINRAG_QDRANT_URL=http://localhost:6333`.
Index folders built by an older version must be rebuilt (`finrag demo` again).

**Better embeddings.** The default LSA embedder is a weak offline baseline. For real use: `ollama pull nomic-embed-text` and set `FINRAG_EMBEDDER=ollama:nomic-embed-text`
(or `pip install -e ".[st]"` and `st:BAAI/bge-small-en-v1.5`), then rebuild the index.

**Speed.** On CPU an 8B model takes tens of seconds per answer. Use `FINRAG_LLM=ollama:llama3.2:latest` for rehearsal, set `OLLAMA_KEEP_ALIVE=1h`, and ask one question before a live demo so the model is loaded.

## Using your own filings

Start `finrag serve`, then use **Add a filing** in the left panel to upload a 10-K, 10-Q or 8-K (.htm or .pdf). Company, form and period are read from the file; fix them under "Details" if needed. Name the company in your question so the right filing is searched.

Bulk loading:

- EDGAR: `SEC_USER_AGENT="Your Name you@uni.edu"`, then `finrag fetch --tickers AAPL MSFT --limit 2 --out edgar_data` and `finrag ingest --manifest edgar_data/manifest.json`.
- FinanceBench: `pip install -e ".[pdf]"`, `finrag financebench --dest financebench_data`, then `finrag ingest` and `finrag eval --gold ...`.

Step-by-step real-data instructions are in `docs/REAL_DATA_RUNBOOK.md`.

## Project layout

| Area | Paths |
|---|---|
| Shared contracts, ingestion, sample data | `src/finrag/core`, `src/finrag/ingestion`, `sample_data`, `pyproject.toml` |
| Retrieval (tokenizer, BM25 reference, embedders, Qdrant index, fusion, filters) | `src/finrag/retrieval` |
| Query rewriting, generation, attribution | `src/finrag/query`, `src/finrag/generation` |
| DSPy program and optimiser | `src/finrag/optimization` |
| Pipeline, evaluation, CLI, server, UI | `src/finrag/pipeline.py`, `cli.py`, `evaluation`, `server` |
| Tests | `tests/` (one folder per area) |

Everything imports the dataclasses in `finrag/core/schema.py`. Per-area notes are in `docs/PART_*.md`.

## Results


### On the built-in corpus (38 questions, k=5, heuristic judge)

The sample corpus is generated by `sample_data/generate.py`: two fictional companies, several filings each, and a gold question set written by the same script. `finrag demo` builds it so the system can be run and tested offline. Qdrant store, LSA embedder, no LLM (extractive answers).

| Setup | Recall@3 | MRR | Answer correct |
|---|---|---|---|
| dense | 74% | 0.51 | 62% |
| sparse (BM25) | 94% | 0.74 | 68% |
| hybrid | 91% | 0.63 | 65% |
| dense + rewrite | 97% | 0.85 | 94% |
| sparse + rewrite | 100% | 0.90 | 91% |
| hybrid + rewrite | 97% | 0.88 | 91% |

How to read this:

- Query rewriting gives the large gain, and it is statistically clear.
- Hybrid beats dense on recall@3 (paired bootstrap interval excludes 0).
- Hybrid does **not** beat BM25 here: BM25 has the higher MRR. Questions about numbers favour keyword matching, and the LSA embedder is weak. Do not claim hybrid wins until a neural embedder has been tried on real filings.
- The gold questions were written by the same script that generated the filings, so these numbers show the machinery works, not how it performs on real filings.
- Qdrant orders exact score ties arbitrarily and the corpus contains many identical boilerplate chunks, so a few rankings can differ between runs or from the `numpy` store (for example hybrid + rewrite answer-correct was 91% on Qdrant and 97% on the `numpy` store, from tie order in 2 of 38 questions).

