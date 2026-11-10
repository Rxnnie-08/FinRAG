import numpy as np
import pytest

from finrag.retrieval import build_index, load_index, save_index
from finrag.retrieval.vectorstore import QdrantIndex

QUERIES = ["total assets", "capital expenditures", "material weakness inventory", "debt maturing 2027", "single-source suppliers"]


@pytest.fixture(scope="module")
def numpy_index(chunks):
    return build_index(chunks, "lsa", vector_store="numpy")


def test_qdrant_is_the_default_store(retriever):
    assert isinstance(retriever.dense, QdrantIndex) and retriever.dense.n == len(retriever.chunks)


def test_qdrant_dense_agrees_with_numpy(numpy_index, retriever):
    # Same embeddings, same cosine metric: neighbours and scores should match. Compare only chunks with similarity > 0.05,
    # because LSA gives many chunks ~0 similarity and their order is arbitrary.
    for q in QUERIES:
        a, b = dict(numpy_index.dense.search(q, 10)), dict(retriever.dense.search(q, 10))
        strong = {i for i, v in a.items() if v > 0.05}
        assert strong and strong <= set(b), q
        assert all(abs(a[i] - b[i]) < 1e-3 for i in strong), q


def test_qdrant_payload_carries_chunk_metadata(retriever):
    pts, _ = retriever.dense.client.scroll(retriever.dense.collection, limit=3, with_payload=True)
    assert {"ticker", "filing_type", "fiscal_year", "fiscal_period", "chunk_id"} <= set(pts[0].payload)


def test_qdrant_respects_allowed_set(retriever):
    allowed = {i for i, c in enumerate(retriever.chunks) if c.ticker == "HLFC"}
    res = retriever.dense.search("total assets", 8, allowed)
    assert res and all(i in allowed for i, _ in res)
    assert retriever.dense.search("total assets", 8, set()) == []


def test_hybrid_filters_and_relaxation_through_qdrant(retriever):
    res = retriever.search("total assets", 5, "hybrid", {"ticker": ["HLFC"], "fiscal_year": [2024]})
    assert res and all(r.chunk.ticker == "HLFC" for r in res if not r.outside_filter)
    relaxed = retriever.search("total assets", 5, "hybrid", {"ticker": ["NWRX"], "fiscal_year": [1999]})
    assert relaxed and all(r.outside_filter for r in relaxed)


def test_persist_and_reload(chunks, tmp_path):
    r = build_index(chunks, "lsa", persist_dir=tmp_path / "qdrant")
    save_index(r, tmp_path)
    r.dense.close()
    r2 = load_index(tmp_path)
    assert isinstance(r2.dense, QdrantIndex) and r2.dense.n == len(chunks)
    fresh = build_index(chunks, "lsa")
    for mode in ("dense", "sparse"):
        assert [x.chunk.chunk_id for x in r2.search("capital expenditures", 5, mode)] == [x.chunk.chunk_id for x in fresh.search("capital expenditures", 5, mode)]
    assert len({x.chunk.chunk_id for x in r2.search("capital expenditures", 5)} & {x.chunk.chunk_id for x in fresh.search("capital expenditures", 5)}) >= 4
    r2.dense.close()


def test_saving_an_in_memory_index_copies_it_to_disk(chunks, tmp_path):
    mem = build_index(chunks, "lsa")                       # no persist_dir: in-memory Qdrant
    save_index(mem, tmp_path)
    assert (tmp_path / "qdrant").exists()
    disk = load_index(tmp_path)
    for mode in ("dense", "sparse"):
        assert [x.chunk.chunk_id for x in disk.search("total assets", 5, mode)] == [x.chunk.chunk_id for x in mem.search("total assets", 5, mode)]
    assert len({x.chunk.chunk_id for x in disk.search("total assets", 5)} & {x.chunk.chunk_id for x in mem.search("total assets", 5)}) >= 4
    disk.dense.close()


def test_second_process_gets_a_clear_error(chunks, tmp_path):
    r = build_index(chunks, "lsa", persist_dir=tmp_path / "qdrant")
    with pytest.raises(RuntimeError, match="FINRAG_QDRANT_URL"):
        QdrantIndex.attach(r.dense.embedder, tmp_path / "qdrant")
    r.dense.close()


def test_numpy_store_still_available(numpy_index, tmp_path):
    save_index(numpy_index, tmp_path)
    assert (tmp_path / "dense.npy").exists() and not (tmp_path / "qdrant").exists()
    assert np.allclose(load_index(tmp_path).dense.matrix, numpy_index.dense.matrix)


# ---- Qdrant runs sparse and hybrid search itself -------------------------------------------------------------------
def test_qdrant_sparse_equals_in_house_bm25(retriever):
    """Qdrant's sparse+IDF search must reproduce Okapi BM25 (bm25.py) for the same tokens."""
    for q in QUERIES + ["Riverbend", "$5,240 million revenue"]:
        ours = dict(retriever.sparse.search(q, 10))
        theirs = dict(retriever.dense.search_sparse(q, 10))
        top = [i for i, _ in sorted(ours.items(), key=lambda kv: -kv[1])][:5]
        assert set(top[:3]) <= set(theirs), q
        for i in set(ours) & set(theirs):
            assert abs(ours[i] - theirs[i]) <= 1e-3 * max(1.0, ours[i]), (q, i)


def test_qdrant_hybrid_matches_our_rrf(retriever):
    from finrag.retrieval.fusion import reciprocal_rank_fusion
    for q in QUERIES:
        d, s = retriever.dense.search(q, 50), retriever.dense.search_sparse(q, 50)
        ours = {i: sc for i, sc, _ in reciprocal_rank_fusion({"dense": [i for i, _ in d], "sparse": [i for i, _ in s]}, 60)}
        native = dict(retriever.dense.search_hybrid(q, q, 10))
        # Qdrant counts ranks from 0, so its scores sit a constant above ours; the ORDER must agree. Chunks with exactly equal
        # scores (the same boilerplate in two filings) are fused in arbitrary order inside Qdrant, so compare by chunk TEXT.
        text = lambda ids: {retriever.chunks[i].text for i in ids}                       # noqa: E731
        top_ours = sorted(ours, key=lambda i: -ours[i])
        assert text(top_ours[:2]) <= text(top_ours[:6]) and text(sorted(native, key=lambda i: -native[i])[:2]) <= text(top_ours[:6]), q
        assert text(sorted(native, key=lambda i: -native[i])[:1]) == text(top_ours[:1]) or \
            retriever.chunks[sorted(native, key=lambda i: -native[i])[0]].text in text(top_ours[:3]), q


def test_all_three_modes_run_in_qdrant_with_explanations(retriever):
    assert retriever.native_hybrid
    res = retriever.search("capital expenditures fiscal 2024", 5, "hybrid", {"ticker": ["NWRX"]})
    assert len(res) == 5 and all(r.method == "hybrid" and r.chunk.ticker == "NWRX" for r in res)
    assert any(set(r.ranks) == {"dense", "sparse"} for r in res)                  # UI can still say "dense rank 2, keyword rank 1"
    for mode in ("dense", "sparse"):
        r2 = retriever.search("capital expenditures", 4, mode, {"ticker": ["HLFC"]})
        assert r2 and all(x.chunk.ticker == "HLFC" for x in r2 if not x.outside_filter)


def test_results_are_deterministic(retriever):
    a = [r.chunk.chunk_id for r in retriever.search("total assets", 6, "hybrid")]
    assert a == [r.chunk.chunk_id for r in retriever.search("total assets", 6, "hybrid")]


def test_old_index_without_sparse_vectors_is_rejected_clearly(tmp_path):
    from qdrant_client import QdrantClient, models
    from finrag.retrieval.dense import make_embedder
    c = QdrantClient(path=str(tmp_path / "qdrant"))
    c.create_collection("finrag_chunks", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    c.close()
    with pytest.raises(FileNotFoundError, match="older version"):
        QdrantIndex.attach(make_embedder("lsa"), tmp_path / "qdrant")
