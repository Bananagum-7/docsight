"""Reciprocal Rank Fusion.

score(d) = sum over retrievers r of 1 / (k + rank_r(d))

Only ranks are used, never raw scores. BM25 scores are unbounded (0 to 30+)
while cosine similarity lives in a narrow band (~0.5 to 0.9), so adding them
directly would need per-corpus calibration. RRF needs none. k=60 is the value
from the original paper (Cormack et al., 2009); it damps the influence of the
very top ranks so one retriever can't dominate.
"""

from __future__ import annotations

from docsight.models import ScoredChunk


def rrf(result_lists: list[list[ScoredChunk]], k: int = 60, top_n: int | None = None) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    keep: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for rank, sc in enumerate(results, start=1):
            cid = sc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            keep.setdefault(cid, sc)
    fused = [ScoredChunk(chunk=keep[cid].chunk, score=s, source="rrf") for cid, s in scores.items()]
    fused.sort(key=lambda sc: sc.score, reverse=True)
    return fused[:top_n] if top_n else fused
