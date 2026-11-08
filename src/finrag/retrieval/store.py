"""Build / save / load the index.

Layout under <home>/index/:  chunks.jsonl, dense.npy, embedder.pkl (LSA only), meta.json
BM25 is rebuilt on load (it takes milliseconds). embedder.pkl is a pickle of YOUR OWN fitted model: only load
index folders you built yourself.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from finrag.core.schema import Chunk

from .bm25 import BM25Index
from .dense import DenseIndex, LSAEmbedder, make_embedder
from .hybrid import Retriever


def build_index(chunks: list[Chunk], embedder: str = "lsa", candidates: int = 50, rrf_k: int = 60) -> Retriever:
    if not chunks:
        raise ValueError("No chunks to index. Ingest some filings first.")
    texts = [c.index_text for c in chunks]
    dense = DenseIndex(make_embedder(embedder)).fit(texts)
    sparse = BM25Index().fit(texts)
    return Retriever(chunks, dense, sparse, candidates=candidates, rrf_k=rrf_k)


def save_index(r: Retriever, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in r.chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(folder / "dense.npy", r.dense.matrix)
    emb = r.dense.embedder
    if isinstance(emb, LSAEmbedder):
        with open(folder / "embedder.pkl", "wb") as f:
            pickle.dump(emb, f)
    (folder / "meta.json").write_text(json.dumps({"embedder": emb.name, "n_chunks": len(r.chunks),
                                                  "candidates": r.candidates, "rrf_k": r.rrf_k}))


def load_index(folder: str | Path) -> Retriever:
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    chunks = [Chunk.from_dict(json.loads(l)) for l in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
    if meta["embedder"].startswith("lsa"):
        with open(folder / "embedder.pkl", "rb") as f:
            emb = pickle.load(f)
    else:
        emb = make_embedder(meta["embedder"])
    dense = DenseIndex(emb)
    dense.matrix = np.load(folder / "dense.npy")
    sparse = BM25Index().fit([c.index_text for c in chunks])
    return Retriever(chunks, dense, sparse, candidates=meta.get("candidates", 50), rrf_k=meta.get("rrf_k", 60))
