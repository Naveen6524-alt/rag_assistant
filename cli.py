"""Command line interface.

  python cli.py eval                       # evaluate one configuration
  python cli.py sweep                      # compare chunk sizes / heading prefix (BM25)
  python cli.py sweep --dense              # also dense + hybrid (needs sentence-transformers)
  python cli.py ask "How do I enable CORS?"            # retrieval only
  python cli.py ask "How do I enable CORS?" --generate # + LLM answer (needs API key)
"""
import argparse
import csv
import sys
from pathlib import Path

from rag.chunking import load_corpus
from rag.evaluate import evaluate, load_questions
from rag.retrievers import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever


def build_retriever(kind, chunks, prepend_heading=True, model="BAAI/bge-small-en-v1.5", rerank=False):
    if kind == "bm25":
        r = BM25Retriever(chunks, prepend_heading)
    elif kind == "dense":
        r = DenseRetriever(chunks, model, prepend_heading)
    elif kind == "hybrid":
        r = HybridRetriever([BM25Retriever(chunks, prepend_heading),
                             DenseRetriever(chunks, model, prepend_heading)])
    else:
        raise ValueError(kind)
    return CrossEncoderReranker(r, chunks, prepend_heading=prepend_heading) if rerank else r


def add_common(p):
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--chunk-words", type=int, default=180)
    p.add_argument("--overlap", type=int, default=30)
    p.add_argument("--no-heading", action="store_true", help="don't prepend heading path to indexed text")
    p.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--rerank", action="store_true")


def fmt(m):
    return (f"file-level: hit@1 {m['file_hit@1']:.2f} | hit@3 {m['file_hit@3']:.2f} | hit@5 {m['file_hit@5']:.2f} "
            f"| MRR {m['file_mrr']:.2f}\nphrase-level (stricter): hit@1 {m['phrase_hit@1']:.2f} | "
            f"hit@5 {m['phrase_hit@5']:.2f} | MRR {m['phrase_mrr']:.2f}")


def cmd_eval(a):
    chunks = load_corpus(a.docs, a.chunk_words, a.overlap)
    r = build_retriever(a.retriever, chunks, not a.no_heading, a.model, a.rerank)
    res = evaluate(r, chunks, load_questions(a.questions))
    print(f"{a.retriever} | chunk_words={a.chunk_words} | heading={'off' if a.no_heading else 'on'} "
          f"| {len(chunks)} chunks | {res['metrics']['n_questions']} questions")
    print(fmt(res["metrics"]))
    out = Path("eval") / f"results_{a.retriever}_{a.chunk_words}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=res["rows"][0].keys()); w.writeheader(); w.writerows(res["rows"])
    print(f"\nPer-question results -> {out}")
    misses = [x for x in res["rows"] if x["file_rank"] is None or x["file_rank"] > 1]
    print(f"\nQuestions whose right file was NOT at rank 1 ({len(misses)}): do your error analysis here")
    for x in misses:
        print(f"  file_rank={x['file_rank']}  Q: {x['question']}\n      gold={x['gold_files']}  top1={x['top1_source']} > {x['top1_heading']}")


def cmd_sweep(a):
    questions = load_questions(a.questions)
    kinds = ["bm25"] + (["dense", "hybrid"] if a.dense else [])
    print("file-level metrics (phrase-level hit@5 in the last column)")
    print(f"{'retriever':<8} {'chunk':>5} {'heading':>7} {'#chunks':>7} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>5} {'ph@5':>5}")
    for cw in (80, 180, 350):
        chunks = load_corpus(a.docs, cw, a.overlap)
        for heading in (True, False):
            for kind in kinds:
                m = evaluate(build_retriever(kind, chunks, heading, a.model), chunks, questions)["metrics"]
                print(f"{kind:<8} {cw:>5} {'on' if heading else 'off':>7} {len(chunks):>7} "
                      f"{m['file_hit@1']:>6.2f} {m['file_hit@3']:>6.2f} {m['file_hit@5']:>6.2f} {m['file_mrr']:>5.2f} {m['phrase_hit@5']:>5.2f}")


def cmd_ask(a):
    chunks = load_corpus(a.docs, a.chunk_words, a.overlap)
    r = build_retriever(a.retriever, chunks, not a.no_heading, a.model, a.rerank)
    hits = [chunks[i] for i, _ in r.search(a.question, a.k)]
    for n, c in enumerate(hits, 1):
        print(f"[{n}] {c.source} > {c.heading}\n    {c.text[:200].replace(chr(10), ' ')}...\n")
    if a.generate:
        from rag.generate import answer
        print("ANSWER:\n" + answer(a.question, hits))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval"); add_common(e); e.add_argument("--questions", default="eval/questions.jsonl"); e.set_defaults(fn=cmd_eval)
    s = sub.add_parser("sweep"); add_common(s); s.add_argument("--questions", default="eval/questions.jsonl")
    s.add_argument("--dense", action="store_true"); s.set_defaults(fn=cmd_sweep)
    q = sub.add_parser("ask"); add_common(q); q.add_argument("question"); q.add_argument("-k", type=int, default=4)
    q.add_argument("--generate", action="store_true"); q.set_defaults(fn=cmd_ask)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
