"""Typed data objects shared by every stage of the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable piece of documentation, with enough metadata to cite it."""

    id: str
    text: str  # what gets embedded and BM25-indexed (includes the breadcrumb header)
    doc_path: str  # e.g. "tutorial/background-tasks.md"
    title: str  # page title
    breadcrumb: str  # "Background Tasks > Using BackgroundTasks"
    url: str  # deep link with #anchor
    has_code: bool = False


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""


class Citation(BaseModel):
    n: int
    title: str
    breadcrumb: str
    url: str
    doc_path: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""
    verification: dict = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
