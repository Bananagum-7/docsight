"""HTTP API.

    uvicorn docsight.api:app --reload
    curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question": "How do background tasks work?"}'

The pipeline (models + index) is loaded once at startup via the lifespan
handler, not per request: loading the reranker takes seconds.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from docsight.models import Answer
from docsight.tracing import flush

log = logging.getLogger("docsight")
state: dict = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "pipeline" not in state:  # tests inject their own pipeline
        from docsight.pipeline import load_pipeline

        state["pipeline"] = load_pipeline()
    yield
    flush()


app = FastAPI(title="DocSight", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": "pipeline" in state}


@app.post("/ask", response_model=Answer)
async def ask(req: AskRequest) -> Answer:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "pipeline not loaded")
    # The pipeline is synchronous (CPU-bound reranker + blocking HTTP to Groq).
    # Running it in a worker thread keeps the event loop free for other requests.
    try:
        return await asyncio.to_thread(pipeline.ask, req.question)
    except Exception as exc:
        log.exception("ask failed")
        raise HTTPException(502, f"upstream error: {exc.__class__.__name__}") from exc


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest) -> dict:
    """Retrieval only, no LLM call. Useful for debugging and for the eval."""
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "pipeline not loaded")
    res = await asyncio.to_thread(pipeline.retriever.retrieve, req.question, req.top_k)
    return {
        "results": [{"score": sc.score, "breadcrumb": sc.chunk.breadcrumb, "url": sc.chunk.url, "text": sc.chunk.text} for sc in res.chunks],
        "timings_ms": res.timings_ms,
    }
