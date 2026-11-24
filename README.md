# RAG assistant with a retrieval evaluation harness

A question-answering assistant over a real documentation set (the FastAPI tutorial, 51 markdown files),
built so that **retrieval quality is measured, not assumed**.

- Markdown-aware chunking (heading-aware, paragraph packing, overlap)
- Three retrievers: BM25, dense embeddings, and hybrid (reciprocal rank fusion), plus an optional cross-encoder reranker
- An evaluation harness: hit@k and MRR against hand-written questions, with per-question CSV output for error analysis
- Optional LLM answer step that must cite the passages it used
- Streamlit UI that always shows the retrieved passages next to the answer

## Quick start
```
pip install -r requirements.txt
python cli.py eval                 # one configuration, with a list of questions it got wrong
python cli.py sweep                # chunk size x heading prefix (BM25)
python cli.py ask "How do I enable CORS?"
streamlit run app.py
python -m pytest tests -q
```
Dense / hybrid / rerank need `pip install sentence-transformers` (pulls in PyTorch, use Python 3.11 or 3.12).
Answer generation needs `pip install anthropic` and the `ANTHROPIC_API_KEY` environment variable.

## What was measured (BM25 only, 20 starter questions)
File-level = "a retrieved chunk came from a gold file". Phrase-level is stricter (see Metrics below).

| chunk words | heading prefix | #chunks | hit@1 | hit@3 | hit@5 | MRR | phrase hit@5 |
|---|---|---|---|---|---|---|---|
| 80  | on  | 931 | 0.35 | 0.50 | 0.65 | 0.46 | 0.20 |
| 80  | off | 931 | 0.40 | 0.50 | 0.75 | 0.51 | 0.30 |
| 180 | on  | 543 | 0.40 | 0.55 | 0.80 | 0.52 | 0.30 |
| 180 | off | 543 | 0.40 | 0.60 | 0.75 | 0.51 | 0.35 |
| 350 | on  | 498 | 0.35 | 0.60 | 0.80 | 0.49 | 0.30 |
| 350 | off | 498 | 0.35 | 0.60 | 0.75 | 0.50 | 0.35 |

**How to read this honestly:** with 20 questions, one question is worth 0.05. Every difference between
rows above is one or two questions, so **none of these configurations is demonstrably better than another**.
The only reasonably clear signal is that very small chunks (80 words) are a bit worse. The BM25 baseline
finds the right page in its top 5 about 75-80% of the time but ranks it first only ~40% of the time.
Dense / hybrid / reranked results are **not** in this table because they have not been run yet.

## Metrics
- **file-level hit@k / MRR**: coarse but consistent. Headline metric.
- **phrase-level hit@k**: the chunk must come from a gold file *and* contain a gold phrase. Stricter, but uneven:
  for some questions the phrase is in 1 of 8 chunks of the file, for others in every chunk. Secondary signal only.
- Gold labels are files and phrases, not chunk IDs, so they stay valid when the chunk size changes.

## Known limitations
- The corpus has the prose of the docs but **not the code samples** (those live in a separate folder upstream), so questions must be about concepts.
- The 20 starter questions were written after looking at the docs, so they are easier than questions written cold.
  Write your own set (aim for 40-50, including a few that are ambiguous or span two pages).
- Retrieval evaluation does not measure answer quality. Faithfulness of generated answers is not evaluated yet.
- Not run in the environment where this was scaffolded (no model downloads or API key available):
  dense retrieval, hybrid, reranker, and the LLM answer step. Unit tests cover their mechanics with a stub embedder only.
- `max_words` does not count the overlap words carried into the next chunk, so chunks can reach about `max_words + overlap`.

## Decisions I made and why  (write these yourself, in your own words)
1. Why chunk by headings first, then by size?
2. What chunk size did I pick, and what evidence do I have for it?
3. BM25 vs dense vs hybrid: which won on my questions, and what does the error analysis show?
4. Why reciprocal rank fusion instead of adding normalised scores?
5. What did the heading-prefix ablation show, and is the difference bigger than the noise?
6. How would I evaluate answer faithfulness, not just retrieval?
7. What would I change with 10x more documents?

## Data
Docs copied from https://github.com/fastapi/fastapi (MIT licence); see data/docs/ATTRIBUTION.md.txt.
