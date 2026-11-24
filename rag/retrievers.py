"""Retrievers: BM25 (lexical), Dense (embeddings), Hybrid (rank fusion), optional reranker.

All retrievers expose:  search(query, k) -> list[(chunk_index, score)]
"""
import hashlib
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

STOP = set("""a an and are as at be by for from has have how i in is it its of on or that the
this to was were what when where which who will with you your do does can my should""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOP]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks, prepend_heading: bool = True):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.index_text(prepend_heading)) for c in chunks])

    def search(self, query: str, k: int = 5):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    """Embedding retriever. Needs `sentence-transformers` (pulls in PyTorch).

    `embedder` can be injected for tests; it must have .encode(list[str]) -> np.ndarray.
    Embeddings are cached on disk so you only pay the encoding cost once per config.
    """
    name = "dense"

    def __init__(self, chunks, model_name: str = "BAAI/bge-small-en-v1.5",
                 prepend_heading: bool = True, embedder=None, cache_dir: str = ".cache"):
        self.chunks = chunks
        self.model_name = model_name
        self.embedder = embedder or self._load_model(model_name)
        texts = [c.index_text(prepend_heading) for c in chunks]
        self.matrix = self._embed_cached(texts, cache_dir if embedder is None else None)

    @staticmethod
    def _load_model(model_name):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Dense retrieval needs sentence-transformers (and PyTorch). "
                "Install it in a Python 3.11/3.12 environment: pip install sentence-transformers"
            ) from e
        return SentenceTransformer(model_name)

    def _encode(self, texts):
        try:
            vecs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except TypeError:  # injected test embedders may not accept those kwargs
            vecs = self.embedder.encode(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    def _embed_cached(self, texts, cache_dir):
        if cache_dir is None:
            return self._encode(texts)
        key = hashlib.sha1((self.model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m = self._encode(texts)
        np.save(path, m)
        return m

    def search(self, query: str, k: int = 5):
        q = self._encode([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class HybridRetriever:
    """Reciprocal Rank Fusion: score(d) = sum over retrievers of 1 / (rrf_k + rank).

    RRF needs no score normalisation, which is why it's the usual first choice for
    mixing BM25 scores with cosine similarities (they live on different scales).
    """
    name = "hybrid"

    def __init__(self, retrievers, rrf_k: int = 60, pool: int = 30):
        self.retrievers, self.rrf_k, self.pool = retrievers, rrf_k, pool

    def search(self, query: str, k: int = 5):
        fused: dict[int, float] = {}
        for r in self.retrievers:
            for rank, (idx, _) in enumerate(r.search(query, self.pool), start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(i, s) for i, s in top]


class CrossEncoderReranker:
    """Optional second stage: re-score the top candidates with a cross-encoder.

    Wraps any retriever. Needs sentence-transformers. NOT exercised in the sandbox
    this project was written in, so run `python cli.py eval --rerank` yourself and
    check the numbers before claiming anything about it.
    """
    def __init__(self, base, chunks, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 pool: int = 20, prepend_heading: bool = True):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("Reranking needs sentence-transformers (see DenseRetriever).") from e
        self.base, self.chunks, self.pool = base, chunks, pool
        self.prepend_heading = prepend_heading
        self.model = CrossEncoder(model_name)
        self.name = f"{base.name}+rerank"

    def search(self, query: str, k: int = 5):
        cands = [i for i, _ in self.base.search(query, self.pool)]
        pairs = [(query, self.chunks[i].index_text(self.prepend_heading)) for i in cands]
        scores = self.model.predict(pairs)
        order = np.argsort(-np.asarray(scores))[:k]
        return [(cands[j], float(scores[j])) for j in order]
