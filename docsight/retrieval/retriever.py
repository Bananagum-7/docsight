"""Hybrid retriever: (BM25 + dense) -> RRF -> rerank.

Every stage can be switched off so the eval script can measure what each one
contributes (the ablation table in the README):

    mode="bm25"     lexical only
    mode="dense"    semantic only
    mode="hybrid"   both, fused with Reciprocal Rank Fusion
    reranker        NoopReranker disables the cross-encoder
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from docsight.models import ScoredChunk
from docsight.retrieval.fusion import rrf
from docsight.retrieval.indexes import BM25Index, DenseIndex
from docsight.retrieval.rerank import NoopReranker, Reranker

Mode = Literal["bm25", "dense", "hybrid"]


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex | None = None,
        reranker: Reranker | None = None,
        mode: Mode = "hybrid",
        candidates: int = 30,
        fused_top_k: int = 30,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ):
        if mode != "bm25" and dense is None:
            raise ValueError(f"mode={mode!r} needs a dense index")
        self.bm25, self.dense = bm25, dense
        self.reranker = reranker or NoopReranker()
        self.mode = mode
        self.candidates, self.fused_top_k, self.final_top_k, self.rrf_k = candidates, fused_top_k, final_top_k, rrf_k

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        t: dict[str, float] = {}
        start = time.perf_counter()
        lists: list[list[ScoredChunk]] = []
        if self.mode in ("bm25", "hybrid"):
            lists.append(self.bm25.search(query, self.candidates))
        if self.mode in ("dense", "hybrid"):
            lists.append(self.dense.search(query, self.candidates))
        t["search"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        fused = rrf(lists, k=self.rrf_k, top_n=self.fused_top_k) if len(lists) > 1 else lists[0][: self.fused_top_k]
        t["fuse"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        final = self.reranker.rerank(query, fused, top_k or self.final_top_k)
        t["rerank"] = (time.perf_counter() - start) * 1000
        return RetrievalResult(chunks=final, timings_ms=t)
