"""Build / save / load the index.

Layout under <home>/index/:  chunks.jsonl, dense.npy (numpy store) or qdrant/ (Qdrant store), embedder.pkl (LSA only), meta.json
BM25 is rebuilt on load (it takes milliseconds). embedder.pkl is a pickle of YOUR OWN fitted model: only load
index folders you built yourself.
"""
from __future__ import annotations

import json
import shutil
import pickle
from pathlib import Path

import numpy as np

from finrag.core.schema import Chunk

from .bm25 import BM25Index
from .dense import DenseIndex, LSAEmbedder, make_embedder, ollama_embed_problem, resolve_embedder
from .hybrid import Retriever


def build_index(chunks: list[Chunk], embedder: str = "lsa", candidates: int = 50, rrf_k: int = 60,
                vector_store: str = "qdrant", persist_dir: str | Path | None = None) -> Retriever:
    """vector_store: "qdrant" (default; Qdrant vector database, local folder persist_dir or a server) or "numpy" (in-memory matrix)."""
    if not chunks:
        raise ValueError("No chunks to index. Ingest some filings first.")
    texts = [c.index_text for c in chunks]
    sparse = BM25Index().fit(texts)
    emb, note = resolve_embedder(embedder)
    if vector_store == "qdrant":
        from .vectorstore import QdrantIndex, payload_of, server_url
        if persist_dir is not None and not server_url() and Path(persist_dir).exists():
            # A rebuild starts from an empty folder. Reusing an old local collection (for example one built with a different
            # vector size) fails with a numpy "could not broadcast" error, so never rely on deleting it in place.
            try:
                shutil.rmtree(persist_dir)
            except OSError as e:
                raise RuntimeError(f"Could not clear the old index folder {persist_dir} ({e}). Stop `finrag serve` or any other finrag "
                                   "process that uses it, or delete the folder by hand, then try again.") from e
        dense = QdrantIndex(emb, persist_dir).fit(texts, [payload_of(c) for c in chunks], sparse)
    elif vector_store == "numpy":
        dense = DenseIndex(emb).fit(texts)
    else:
        raise ValueError(f"Unknown vector store '{vector_store}'. Use qdrant | numpy.")
    r = Retriever(chunks, dense, sparse, candidates=candidates, rrf_k=rrf_k)
    r.notes = [note] if note else []
    return r


def save_index(r: Retriever, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in r.chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    store = getattr(r.dense, "store_name", "numpy")
    if store == "numpy":
        np.save(folder / "dense.npy", r.dense.matrix)
    elif not r.dense.server and (r.dense.path is None or Path(r.dense.path).resolve() != (folder / "qdrant").resolve()):
        r.dense.export(folder / "qdrant").close()           # in-memory index: copy its vectors to disk
    emb = r.dense.embedder
    if isinstance(emb, LSAEmbedder):
        with open(folder / "embedder.pkl", "wb") as f:
            pickle.dump(emb, f)
    (folder / "meta.json").write_text(json.dumps({"embedder": emb.name, "n_chunks": len(r.chunks),
                                                  "candidates": r.candidates, "rrf_k": r.rrf_k, "vector_store": store}), encoding="utf-8")


def load_index(folder: str | Path) -> Retriever:
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(json.loads(l)) for l in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
    if meta["embedder"].startswith("lsa"):
        with open(folder / "embedder.pkl", "rb") as f:
            emb = pickle.load(f)
    else:
        if (problem := ollama_embed_problem(meta["embedder"])):
            raise RuntimeError(f"This index was built with {meta['embedder']} and questions must be embedded with the same model: {problem}. "
                               "Start Ollama, or rebuild the index with FINRAG_EMBEDDER=lsa.")
        emb = make_embedder(meta["embedder"])
    if meta.get("vector_store") in ("qdrant",):
        from .vectorstore import QdrantIndex
        dense = QdrantIndex.attach(emb, folder / "qdrant")
    else:
        dense = DenseIndex(emb)
        dense.matrix = np.load(folder / "dense.npy")
    sparse = BM25Index().fit([c.index_text for c in chunks])
    return Retriever(chunks, dense, sparse, candidates=meta.get("candidates", 50), rrf_k=meta.get("rrf_k", 60))
