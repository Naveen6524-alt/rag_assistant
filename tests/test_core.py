import zlib

import numpy as np

from rag.chunking import Chunk, pack_paragraphs, split_sections
from rag.evaluate import evaluate
from rag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, tokenize

MD = """# Page Title { #page-title }

Intro paragraph about the page.

{* ../../docs_src/x/tutorial001.py hl[1,2] *}

## Section A { #a }

/// tip | A tip

Alpha text mentions widgets.

///

```python
# this is a comment, not a heading
x = 1
```

### Sub A1

Deeper text about gadgets.
"""


def test_sections_strip_noise_and_track_heading_path():
    secs = list(split_sections(MD))
    paths = [p for p, _ in secs]
    assert paths == ["Page Title", "Page Title > Section A", "Page Title > Section A > Sub A1"]
    joined = "\n".join(b for _, b in secs)
    assert "docs_src" not in joined          # include directive removed
    assert "{ #" not in joined               # anchors removed
    assert "///" not in joined               # admonition fences removed
    assert "# this is a comment" in joined   # '#' inside a code fence is NOT a heading


def test_pack_paragraphs_respects_size_and_drops_nothing():
    text = "\n\n".join(" ".join(f"w{i}_{j}" for j in range(40)) for i in range(6))
    chunks = pack_paragraphs(text, max_words=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 + 10 for c in chunks)   # max_words + overlap seed
    assert "w5_39" in chunks[-1] and "w0_0" in chunks[0]
    assert pack_paragraphs("", 100, 10) == []


def test_pack_paragraphs_splits_one_giant_paragraph():
    giant = " ".join(f"t{i}" for i in range(500))
    chunks = pack_paragraphs(giant, max_words=100, overlap=20)
    assert len(chunks) >= 5 and all(len(c.split()) <= 100 for c in chunks)


def _tiny_corpus():
    texts = {
        "a.md": ("Cats", "Cats purr and nap in the sun all day"),
        "b.md": ("Dogs", "Dogs bark and fetch sticks at the park"),
        "c.md": ("Fish", "Fish swim in the ocean and eat plankton"),
    }
    return [Chunk(id=f"{k}::0", source=k, heading=h, text=t) for k, (h, t) in texts.items()]


def test_bm25_ranks_the_lexically_matching_chunk_first():
    chunks = _tiny_corpus()
    hits = BM25Retriever(chunks).search("which animal likes to bark and fetch", k=3)
    assert chunks[hits[0][0]].source == "b.md"


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("How do I run The Server?") == ["run", "server"]


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. Tests MECHANICS only (shapes, ordering,
    normalisation) - it is lexical, so it says nothing about real semantic quality."""
    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                out[r, zlib.crc32(tok.encode()) % 64] += 1.0
        return out


def test_dense_retriever_mechanics_with_stub_embedder():
    chunks = _tiny_corpus()
    d = DenseRetriever(chunks, embedder=HashEmbedder())
    assert np.allclose(np.linalg.norm(d.matrix, axis=1), 1.0, atol=1e-5)
    hits = d.search("fish in the ocean", k=3)
    assert chunks[hits[0][0]].source == "c.md"
    assert hits[0][1] >= hits[1][1] >= hits[2][1]


def test_hybrid_rrf_prefers_documents_both_retrievers_like():
    class Fixed:
        def __init__(self, order): self.order = order
        def search(self, q, k=5): return [(i, 1.0) for i in self.order[:k]]
    # doc 2 is 2nd in both lists, doc 0 is 1st in one only, doc 9 is 1st in the other only
    h = HybridRetriever([Fixed([0, 2, 5]), Fixed([9, 2, 6])], rrf_k=60)
    assert h.search("q", 3)[0][0] == 2


def test_evaluate_metric_math_on_a_fake_retriever():
    chunks = _tiny_corpus()

    class Fake:
        def __init__(self, ranking): self.ranking = ranking
        def search(self, q, k=5): return [(i, 1.0) for i in self.ranking[q][:k]]

    questions = [
        {"question": "q1", "gold_files": ["a.md"], "gold_phrase": "purr"},   # found at rank 1
        {"question": "q2", "gold_files": ["b.md"], "gold_phrase": "bark"},   # found at rank 2
        {"question": "q3", "gold_files": ["c.md"], "gold_phrase": "plankton"},  # not retrieved
    ]
    ranking = {"q1": [0, 1, 2], "q2": [0, 1, 2], "q3": [0, 1]}
    m = evaluate(Fake(ranking), chunks, questions, ks=(1, 3))["metrics"]
    assert m["file_hit@1"] == 1 / 3
    assert abs(m["file_hit@3"] - 2 / 3) < 1e-9
    assert abs(m["file_mrr"] - (1 + 0.5 + 0) / 3) < 1e-9
    assert m["phrase_hit@1"] == 1 / 3
