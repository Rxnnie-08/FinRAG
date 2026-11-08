"""Dense retrieval: pluggable embedders behind one interface.

  lsa                       TF-IDF + truncated SVD, fitted on the corpus. Offline and instant; a *weak* semantic
                            baseline, which is exactly why it is the default for tests and demos.
  st:<model>                sentence-transformers, e.g. st:BAAI/bge-small-en-v1.5 (recommended for real runs)
  ollama:<model>            local Ollama server embeddings, e.g. ollama:nomic-embed-text
"""
from __future__ import annotations

import numpy as np


class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class LSAEmbedder(Embedder):
    name = "lsa"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.tfidf = None
        self.svd = None

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1,
                                     token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")
        X = self.tfidf.fit_transform(texts)
        dim = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=dim, random_state=0)
        self.svd.fit(X)
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return _l2(self.svd.transform(self.tfidf.transform(texts)))


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("pip install sentence-transformers to use st:<model> embedders") from e
        self.name = f"st:{model}"
        self.model_name = model
        self.model = SentenceTransformer(model)
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model.lower() else (
            "query: " if "e5" in model.lower() else "")
        self.doc_prefix = "passage: " if "e5" in model.lower() else ""

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        pre = self.query_prefix if is_query else self.doc_prefix
        return _l2(np.asarray(self.model.encode([pre + t for t in texts], batch_size=32, show_progress_bar=False)))


class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.name = f"ollama:{model}"
        self.model, self.base = model, base_url.rstrip("/")

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        import requests
        r = requests.post(f"{self.base}/api/embed", json={"model": self.model, "input": texts}, timeout=300)
        r.raise_for_status()
        return _l2(np.asarray(r.json()["embeddings"], dtype="float32"))


def _l2(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype="float32")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-9)


def make_embedder(spec: str) -> Embedder:
    kind, _, arg = spec.partition(":")
    if kind == "lsa":
        return LSAEmbedder(int(arg) if arg.isdigit() else 256)
    if kind == "st":
        return SentenceTransformerEmbedder(arg or "BAAI/bge-small-en-v1.5")
    if kind == "ollama":
        return OllamaEmbedder(arg or "nomic-embed-text")
    raise ValueError(f"Unknown embedder '{spec}'. Use lsa | st:<model> | ollama:<model>")


class DenseIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.matrix: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "DenseIndex":
        self.embedder.fit(texts)
        batches = [self.embedder.encode(texts[i:i + 64]) for i in range(0, len(texts), 64)]
        self.matrix = np.vstack(batches) if batches else np.zeros((0, 1), dtype="float32")
        return self

    def search(self, query: str, k: int = 50, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if self.matrix is None or not len(self.matrix):
            return []
        q = self.embedder.encode([query], is_query=True)[0]
        sims = self.matrix @ q
        if allowed is not None:
            mask = np.full(len(sims), -np.inf, dtype="float32")
            idx = np.fromiter(allowed, dtype=int)
            mask[idx] = sims[idx]
            sims = mask
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order if np.isfinite(sims[i])]
