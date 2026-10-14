"""The two first-stage retrievers that feed hybrid search.

BM25Index  - lexical. Exact tokens: API names (`BackgroundTasks`), error
             messages, config keys. Dense models often blur these.
DenseIndex - semantic, stored in Qdrant. Paraphrases: "run something after
             the response is sent" ~ "background tasks".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from docsight.models import Chunk, ScoredChunk

# Keep identifiers whole: snake_case, dotted.names, CamelCase all survive.
TOKEN_RE = re.compile(r"[a-z0-9_][a-z0-9_\.]*[a-z0-9_]|[a-z0-9_]")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its my of on or so that the this to was what when where which why with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i]), source="bm25") for i in top if scores[i] > 0]


class DenseIndex:
    COLLECTION = "chunks"

    def __init__(self, embedder, path: Path | None = None):
        from qdrant_client import QdrantClient

        self.embedder = embedder
        self.client = QdrantClient(path=str(path)) if path else QdrantClient(":memory:")
        self._by_point: dict[int, Chunk] = {}

    def build(self, chunks: list[Chunk], batch: int = 64) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.COLLECTION):
            self.client.delete_collection(self.COLLECTION)
        self.client.create_collection(
            self.COLLECTION,
            vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vecs = self.embedder.embed_documents([c.text for c in part])
            self.client.upsert(
                self.COLLECTION,
                points=[
                    models.PointStruct(id=start + i, vector=vecs[i].tolist(), payload={"chunk_id": c.id})
                    for i, c in enumerate(part)
                ],
            )
        self.attach(chunks)

    def attach(self, chunks: list[Chunk]) -> None:
        """Map Qdrant point ids back to chunks (point id = position in chunks.jsonl)."""
        self._by_point = dict(enumerate(chunks))

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        res = self.client.query_points(self.COLLECTION, query=self.embedder.embed_query(query).tolist(), limit=k)
        return [ScoredChunk(chunk=self._by_point[p.id], score=float(p.score), source="dense") for p in res.points]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open() as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
