"""Second-stage reranking.

A bi-encoder (the dense index) embeds query and chunk separately, which is
fast but loses interaction between them. A cross-encoder reads the pair
together ("[CLS] query [SEP] chunk") and outputs one relevance score: much
more accurate, much slower. So we run it only on the ~30 fused candidates.
"""

from __future__ import annotations

from typing import Protocol

from docsight.models import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_chars: int = 2000):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)
        self.max_chars = max_chars

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text[: self.max_chars]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [ScoredChunk(chunk=c.chunk, score=float(s), source="rerank") for c, s in ranked[:top_k]]


class NoopReranker:
    """Keeps fusion order. Used for the 'no rerank' ablation and in tests."""

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]
