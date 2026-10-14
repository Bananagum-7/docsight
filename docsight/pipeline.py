"""End-to-end question answering: retrieve -> (abstain?) -> generate -> verify -> (retry?).

Abstention happens in two places:
- Before generation, if the reranker's best score is too low. Nothing
  relevant was found, so we skip the LLM call entirely (saves tokens).
- After generation, if the model says NOT_IN_DOCS, or if the answer still
  fails verification after one retry with feedback.
"""

from __future__ import annotations

import time
from pathlib import Path

from docsight.config import Settings, settings as default_settings
from docsight.generation.prompts import NOT_FOUND, answer_messages
from docsight.generation.verify import verify
from docsight.llm import LLM
from docsight.models import Answer, Citation, ScoredChunk
from docsight.retrieval.retriever import HybridRetriever
from docsight.tracing import span


class DocSight:
    def __init__(self, retriever: HybridRetriever, llm: LLM, cfg: Settings = default_settings, max_retries: int = 1):
        self.retriever, self.llm, self.cfg, self.max_retries = retriever, llm, cfg, max_retries

    def _citations(self, answer_text: str, chunks: list[ScoredChunk]) -> list[Citation]:
        out = []
        for i, sc in enumerate(chunks, start=1):
            if f"[{i}]" in answer_text:
                c = sc.chunk
                out.append(Citation(n=i, title=c.title, breadcrumb=c.breadcrumb, url=c.url, doc_path=c.doc_path))
        return out

    def ask(self, question: str) -> Answer:
        timings: dict[str, float] = {}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}

        with span("ask", input={"question": question}) as root:
            start = time.perf_counter()
            with span("retrieve"):
                result = self.retriever.retrieve(question)
            timings.update({f"retrieve.{k}": v for k, v in result.timings_ms.items()})
            timings["retrieve"] = (time.perf_counter() - start) * 1000
            chunks = result.chunks

            calibrated = getattr(self.retriever.reranker, "calibrated", False)
            if not chunks or (calibrated and chunks[0].score < self.cfg.min_rerank_score):
                ans = Answer(question=question, answer="I couldn't find this in the documentation.", abstained=True,
                             abstain_reason="no relevant passages retrieved", timings_ms=timings, usage=usage)
                root.update(output=ans.model_dump())
                return ans

            passages = [sc.chunk.text for sc in chunks]
            feedback = None
            text, ver = "", None
            for attempt in range(self.max_retries + 1):
                start = time.perf_counter()
                with span("generate", metadata={"attempt": attempt}):
                    resp = self.llm.complete(answer_messages(question, chunks, feedback), model=self.cfg.answer_model)
                timings[f"generate.{attempt}"] = (time.perf_counter() - start) * 1000
                usage["prompt_tokens"] += resp.usage.get("prompt_tokens", 0)
                usage["completion_tokens"] += resp.usage.get("completion_tokens", 0)
                usage["llm_calls"] += 1
                text = resp.text.strip()

                if NOT_FOUND in text:
                    ans = Answer(question=question, answer="I couldn't find this in the documentation.", abstained=True,
                                 abstain_reason="model reported the answer is not in the passages",
                                 citations=[], timings_ms=timings, usage=usage)
                    root.update(output=ans.model_dump())
                    return ans

                with span("verify"):
                    ver = verify(text, passages)
                if ver.passed:
                    break
                feedback = "; ".join(ver.problems)

            ans = Answer(
                question=question,
                answer=text if ver.passed else "I couldn't produce an answer I could verify against the documentation.",
                citations=self._citations(text, chunks) if ver.passed else [],
                abstained=not ver.passed,
                abstain_reason="" if ver.passed else f"failed verification: {'; '.join(ver.problems)}",
                verification=ver.as_dict(),
                timings_ms=timings,
                usage=usage,
            )
            ans.timings_ms["total"] = sum(v for k, v in timings.items() if k == "retrieve" or k.startswith("generate."))
            root.update(output=ans.model_dump())
            return ans


def load_pipeline(cfg: Settings = default_settings, llm: LLM | None = None, use_reranker: bool = True, mode: str = "hybrid") -> DocSight:
    """Build the production pipeline from a prebuilt index (see scripts/build_index.py)."""
    from docsight.llm import GroqLLM
    from docsight.retrieval.embed import FastEmbedEmbedder
    from docsight.retrieval.indexes import BM25Index, DenseIndex, load_chunks
    from docsight.retrieval.rerank import CrossEncoderReranker

    chunks_path = Path(cfg.index_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No index at {cfg.index_dir}. Run: python scripts/build_index.py")
    chunks = load_chunks(chunks_path)
    dense = None
    if mode != "bm25":
        dense = DenseIndex(FastEmbedEmbedder(cfg.embed_model), path=Path(cfg.index_dir) / "qdrant")
        dense.attach(chunks)
    reranker = CrossEncoderReranker(cfg.rerank_model) if use_reranker else None
    retriever = HybridRetriever(
        BM25Index(chunks), dense, reranker, mode=mode,
        candidates=cfg.candidates_per_retriever, fused_top_k=cfg.fused_top_k, final_top_k=cfg.final_top_k, rrf_k=cfg.rrf_k,
    )
    return DocSight(retriever, llm or GroqLLM(cache_dir=cfg.cache_dir / "llm"), cfg)
