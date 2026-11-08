# Part C: query rewriting, generation, attribution

Owns: `query/` (company directory, fiscal-period resolution, glossary, rewriter) and `generation/` (extractive answerer, LLM
backends, claim verification, citations).

Contract: `QueryRewriter.rewrite(question, history) -> RewrittenQuery` (filters, assumptions, unknown entities);
`verify_answer(text, retrieved)` and `build_citations(...)`. Provenance is deterministic (chunk metadata); the support check
is a heuristic and is reported as a rate.

Test: `pytest tests/c_query_generation` (needs A and B merged for the fixtures). Prompt-engineering material for the course
lives in `generation/generate.py` (`SYSTEM`) and the LLM rewrite prompt in `query/rewriter.py`.
