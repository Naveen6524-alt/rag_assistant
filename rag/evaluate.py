"""Retrieval evaluation: hit@k and MRR against hand-written questions.

Two levels of strictness:
- file-level: a hit if a retrieved chunk comes from any gold file (coarse, forgiving)
- phrase-level: a hit only if the chunk comes from a gold file AND contains any `gold_phrase`
  (stricter, but uneven: if the phrase appears in 1 of 8 chunks it is very strict, if it appears
  in every chunk of the file it equals file-level. Treat it as a secondary signal.)
Gold labels are files + phrases, not chunk IDs, so they survive changes to the chunk size.
"""
import json
from statistics import mean


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(retriever, chunks, questions, ks=(1, 3, 5)):
    kmax = max(ks)
    rows = []
    for q in questions:
        hits = retriever.search(q["question"], kmax)
        gold = set(q["gold_files"])
        raw = q.get("gold_phrase") or []
        phrases = [x.lower() for x in ([raw] if isinstance(raw, str) else raw) if x]
        file_rank = phrase_rank = None
        for rank, (idx, _score) in enumerate(hits, start=1):
            c = chunks[idx]
            if file_rank is None and c.source in gold:
                file_rank = rank
            if (phrase_rank is None and phrases and c.source in gold
                    and any(p in c.text.lower() for p in phrases)):
                phrase_rank = rank
        top = chunks[hits[0][0]] if hits else None
        rows.append({
            "question": q["question"],
            "gold_files": "|".join(sorted(gold)),
            "file_rank": file_rank,
            "phrase_rank": phrase_rank,
            "top1_source": top.source if top else "",
            "top1_heading": top.heading if top else "",
        })

    def rate(key, k):
        return mean(1.0 if r[key] is not None and r[key] <= k else 0.0 for r in rows)

    def mrr(key):
        return mean((1.0 / r[key]) if r[key] else 0.0 for r in rows)

    metrics = {"n_questions": len(rows)}
    for k in ks:
        metrics[f"file_hit@{k}"] = rate("file_rank", k)
        metrics[f"phrase_hit@{k}"] = rate("phrase_rank", k)
    metrics["file_mrr"] = mrr("file_rank")
    metrics["phrase_mrr"] = mrr("phrase_rank")
    return {"metrics": metrics, "rows": rows}
