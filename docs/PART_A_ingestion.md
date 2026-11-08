# Part A: data and ingestion

Owns: `core/` (schema, numbers, fiscal calendar, config), `ingestion/` (HTML tables, chunker, XBRL, PDF, EDGAR client),
`sample_data/` (synthetic filings generator), `pyproject.toml`.

Contract you provide: `ingest_manifest(list[FilingMeta]) -> list[Chunk]`. Tables are atomic or split by rows with the header
repeated; footnotes attach to their table; `context_header` is prepended at index time.

Test: `pytest tests/a_core_ingestion`. Regenerate data: `PYTHONPATH=src python sample_data/generate.py`.
Talking points: colspan/rowspan grid rebuild, detached `$`/`)` merging, unit detection, inline-XBRL facts, SEC fair-access rules
(User-Agent with contact, rate limit).
