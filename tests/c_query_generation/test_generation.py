from finrag.core.schema import Chunk, RetrievedChunk
from finrag.generation.attribution import best_span, build_citations, is_abstention, verify_answer
from finrag.generation.extractive import ABSTAIN, extractive_answer


def rc(text, rank=1, **kw):
    c = Chunk(f"c{rank}", text, "Acme Inc", "ACME", "10-K", 2024, "FY", "2024-12-31", item="Item 8", **kw)
    return RetrievedChunk(c, 1.0, rank)


def test_verify_supported_and_unsupported():
    r = [rc("Total assets were $4,680 million at December 31, 2024.")]
    ok, s = verify_answer("Total assets were $4,680 million [1].", r)
    assert s["support_rate"] == 1.0
    bad, s2 = verify_answer("Total assets were $9,999 million [1].", r)
    assert s2["support_rate"] == 0.0 and bad[0].numeric_ok is False


def test_number_cannot_match_tiny_noise_number():
    # "10-K" in the context header must not validate a wrong $9,999 figure
    from finrag.core.numbers import any_number_match
    assert not any_number_match("10", "$9,999 million") and any_number_match("$1.25 billion", "1,250")


def test_uncited_and_invalid_citation():
    r = [rc("Revenue was $100 million.")]
    claims, s = verify_answer("Revenue was $100 million.", r)
    assert s["citation_coverage"] == 0.0
    _, s2 = verify_answer("Revenue was $100 million [5].", r)
    assert s2["invalid_citations"] == 1


def test_abstention_is_not_a_claim():
    claims, s = verify_answer(f"{ABSTAIN} Not in the corpus.", [rc("x")])
    assert is_abstention(f"{ABSTAIN} foo") and not any(c.supported is False for c in claims)


def test_best_span_picks_matching_sentence():
    span, score = best_span("Net income was $42 million", "Revenue rose. Net income was $42 million in 2024. Other.")
    assert "42" in span and score > 0.5


def test_extractive_prefers_table_row(retriever):
    res = retriever.search("total assets fiscal 2024 Northwind", 6, "hybrid", {"ticker": "NWRX", "fiscal_year": 2024})
    text, _ = extractive_answer("What were total assets?", res)
    assert "4,680" in text


def test_ollama_llm_spec(monkeypatch):
    from finrag.generation import make_llm
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    m = make_llm("ollama:llama3.1:8b", raw=True)
    assert m.model == "llama3.1:8b" and m.base == "http://localhost:11434/v1" and m.name == "ollama:llama3.1:8b"
