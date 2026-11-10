# Part C: query rewriting, generation, attribution

Owns: `query/` (company directory, fiscal-period resolution, glossary, rewriter) and `generation/` (extractive answerer, LLM
backends, claim verification, citations).

Contract: `QueryRewriter.rewrite(question, history) -> RewrittenQuery` (filters, assumptions, unknown entities);
`verify_answer(text, retrieved)` and `build_citations(...)`. Provenance is deterministic (chunk metadata); the support check
is a heuristic and is reported as a rate.

Test: `pytest tests/c_query_generation` (needs A and B merged for the fixtures). Prompt-engineering material for the course
lives in `generation/generate.py` (`SYSTEM`) and the LLM rewrite prompt in `query/rewriter.py`.

DSPy (core): every LLM answer goes through a DSPy program (`optimization/dspy_qa.py`, wired in `generation/llm.py::make_llm`). The signature starts as the hand-written `SYSTEM` prompt; `finrag optimize` runs BootstrapFewShot or MIPROv2 on a train split of the gold questions with the benchmark's own correctness and citation checks, reports baseline vs optimised on a held-out dev split, and saves `<home>/dspy/program.json`, which is loaded automatically. Ollama is the default model; if it is not running the pipeline falls back to extractive answers with a visible note. Tests: `tests/d_eval_app/test_dspy.py`, `test_llm_default.py`. Open item: run it on a real Ollama model and report the dev-split difference whatever its sign.
