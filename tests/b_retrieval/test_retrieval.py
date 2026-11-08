from finrag.retrieval import build_index, load_index, save_index
from finrag.retrieval.bm25 import BM25Index
from finrag.retrieval.fusion import reciprocal_rank_fusion
from finrag.retrieval.tokenizer import tokenize, tokenize_query


def test_tokenizer_keeps_numbers_canonical():
    assert "1250" in tokenize("Total: $1,250 million")
    q = tokenize_query("$1.25 billion")
    assert "1250" in q or "1250000000" in q or "1.25" in q


def test_bm25_ranks_exact_match_first():
    idx = BM25Index().fit(["apple banana", "total assets 4680", "cash 583"])
    assert idx.search("assets 4680", 3)[0][0] == 1
    assert idx.search("assets", 3, allowed={0, 2}) == [] or all(i in (0, 2) for i, _ in idx.search("assets", 3, allowed={0, 2}))


def test_rrf_prefers_docs_in_both_lists():
    out = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 2, 9]})
    assert out[0][0] in (2, 3)
    assert set(out[0][2]) == {"a", "b"}
    assert {d for d, *_ in out} == {1, 2, 3, 9}


def test_all_modes_return_k(retriever):
    for mode in ("dense", "sparse", "hybrid"):
        res = retriever.search("total assets", 4, mode)
        assert len(res) == 4 and [r.rank for r in res] == [1, 2, 3, 4]
        assert all(r.method == mode for r in res)


def test_sparse_finds_exact_number(retriever):
    top = retriever.search("4,680", 3, "sparse")
    assert any("4,680" in r.chunk.text or "4680" in r.chunk.text for r in top)


def test_filters_apply_before_ranking(retriever):
    res = retriever.search("total assets", 5, "hybrid", {"ticker": "HLFC", "fiscal_year": 2024})
    assert res and all(r.chunk.ticker == "HLFC" and r.chunk.fiscal_year == 2024 for r in res if not r.outside_filter)


def test_relaxed_filter_is_flagged(retriever):
    res = retriever.search("total assets", 5, "hybrid", {"ticker": "NWRX", "fiscal_year": 1999})
    assert res and all(r.outside_filter for r in res)


def test_save_load_roundtrip(retriever, tmp_path):
    save_index(retriever, tmp_path)
    r2 = load_index(tmp_path)
    a = [x.chunk.chunk_id for x in retriever.search("capital expenditures", 5, "hybrid")]
    b = [x.chunk.chunk_id for x in r2.search("capital expenditures", 5, "hybrid")]
    assert a == b
