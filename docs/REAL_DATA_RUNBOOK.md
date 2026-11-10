# Real-data runbook (FinanceBench + EDGAR)

Run everything from the project folder with the virtualenv active.

## 0. Install
    pip install -e ".[dev,pdf,st]"          # pdf = pypdf, st = sentence-transformers (neural embeddings)

## 1. FinanceBench (150 questions, 84 filings, ~1 GB of PDFs; first run: try --limit-docs 5)
    finrag financebench --dest fb_data --limit-docs 5
    export FINRAG_EMBEDDER=st:BAAI/bge-small-en-v1.5
    finrag ingest --manifest fb_data/manifest.json --home fb_data/home
    finrag eval --home fb_data/home --gold fb_data/qa_gold.jsonl
Drop `--limit-docs` for the full run (ingest takes a while: ~100 pages per filing, embedded on CPU).

## 2. Answer generation (needed: ~60% of FinanceBench needs arithmetic, the extractive baseline cannot do it)
Option 1, hosted: `pip install -e ".[anthropic]"`, `export ANTHROPIC_API_KEY=...`, `export FINRAG_LLM=anthropic:<model>`.
Option 2, free and local (Ollama): install Ollama, `ollama pull llama3.1:8b`, then `export FINRAG_LLM=ollama:llama3.1:8b`.
Ollama embeddings also work: `ollama pull nomic-embed-text` and `export FINRAG_EMBEDDER=ollama:nomic-embed-text`.
Run the eval with the same variables set. Use a different model for `--judge` than for answering when you can.

## 3. EDGAR HTML (this is where table and footnote preservation is exercised; FinanceBench PDFs flatten tables)
    export SEC_USER_AGENT="Your Name your.email@college.edu"
    finrag fetch --tickers AAPL MSFT NVDA --forms 10-K 10-Q --limit 2 --out edgar_data
    finrag ingest --manifest edgar_data/manifest.json --home edgar_data/home
Write 30-50 gold questions by hand into edgar_data/qa_gold.jsonl (same format as sample_data/qa_gold.jsonl: question, answer,
evidence strings that appear verbatim in the filing, gold_filing). Then `finrag eval --home edgar_data/home`.

## 4. What to put in the report
Take the tables from `<home>/eval/results.md`. Report the paired-bootstrap comparisons as printed, including the "not significant" lines.
Hand-label 30 answers and run `evaluation.judge.agreement` to report judge-vs-human agreement.
Page-level recall is strict (the right number may sit on a neighbouring page); report `filing_hit@k` next to it.
