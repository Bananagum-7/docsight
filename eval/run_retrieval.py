"""Retrieval ablation: how much does each stage help?

    python eval/run_retrieval.py                      # all configs, real models
    python eval/run_retrieval.py --embedder hash --no-rerank   # offline smoke test

Metrics (answerable questions only):
    Hit@k  - share of questions where a gold page appears in the top k chunks
    MRR@10 - mean of 1/rank of the first gold chunk (0 if not in top 10)
    p50/p95 latency of retrieval in ms
No LLM calls, so this runs in seconds and costs nothing.
"""

from __future__ import annotations

import argparse
import time

from common import load_questions, markdown_table, mean, percentile, save

from docsight.config import settings
from docsight.retrieval.indexes import BM25Index, DenseIndex, load_chunks
from docsight.retrieval.retriever import HybridRetriever


def evaluate(retriever: HybridRetriever, questions: list[dict], k_values=(1, 3, 5, 10)) -> dict:
    hits = {k: [] for k in k_values}
    rr, lat = [], []
    misses = []
    for q in questions:
        t0 = time.perf_counter()
        res = retriever.retrieve(q["question"], top_k=max(k_values))
        lat.append((time.perf_counter() - t0) * 1000)
        docs = [sc.chunk.doc_path for sc in res.chunks]
        gold = set(q["gold_docs"])
        first = next((i for i, d in enumerate(docs, start=1) if d in gold), None)
        for k in k_values:
            hits[k].append(1.0 if first and first <= k else 0.0)
        rr.append(1.0 / first if first else 0.0)
        if not first or first > 5:
            misses.append({"id": q["id"], "question": q["question"], "gold": q["gold_docs"], "got": docs[:5]})
    out = {f"hit@{k}": round(mean(v), 3) for k, v in hits.items()}
    out.update({"mrr@10": round(mean(rr), 3), "p50_ms": round(percentile(lat, 0.5), 1), "p95_ms": round(percentile(lat, 0.95), 1)})
    out["misses"] = misses
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["fastembed", "hash"], default="fastembed")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder configs")
    args = ap.parse_args()

    questions = [q for q in load_questions() if q["answerable"]]
    chunks = load_chunks(settings.index_dir / "chunks.jsonl")
    bm25 = BM25Index(chunks)

    if args.embedder == "hash":
        from docsight.retrieval.embed import HashEmbedder

        embedder = HashEmbedder()
    else:
        from docsight.retrieval.embed import FastEmbedEmbedder

        embedder = FastEmbedEmbedder(settings.embed_model)
    dense = DenseIndex(embedder, path=settings.index_dir / "qdrant")
    dense.attach(chunks)

    configs = [
        ("BM25 only", dict(mode="bm25")),
        ("Dense only", dict(mode="dense")),
        ("Hybrid (RRF)", dict(mode="hybrid")),
    ]
    if not args.no_rerank:
        from docsight.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.rerank_model)
        configs += [
            ("Dense + rerank", dict(mode="dense", reranker=reranker)),
            ("Hybrid + rerank", dict(mode="hybrid", reranker=reranker)),
        ]

    rows = []
    for name, kw in configs:
        retriever = HybridRetriever(bm25, dense, candidates=settings.candidates_per_retriever, fused_top_k=settings.fused_top_k, **kw)
        r = evaluate(retriever, questions)
        rows.append({"config": name, **r})
        print(f"{name:18s} hit@1={r['hit@1']:.3f} hit@5={r['hit@5']:.3f} mrr={r['mrr@10']:.3f} p95={r['p95_ms']}ms")

    table = markdown_table(rows, [("config", "Config"), ("hit@1", "Hit@1"), ("hit@3", "Hit@3"), ("hit@5", "Hit@5"),
                                  ("mrr@10", "MRR@10"), ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms")])
    print("\n" + table)
    save("retrieval", {"embedder": args.embedder, "n_questions": len(questions), "configs": rows}, table)


if __name__ == "__main__":
    main()
