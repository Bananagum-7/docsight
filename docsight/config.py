"""Central settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCSIGHT_DATA", ROOT / "data")))
    docs_root: Path = field(default_factory=lambda: Path(os.getenv("DOCSIGHT_DOCS_ROOT", ROOT / "data" / "fastapi" / "docs" / "en" / "docs")))
    code_root: Path = field(default_factory=lambda: Path(os.getenv("DOCSIGHT_CODE_ROOT", ROOT / "data" / "fastapi")))
    base_url: str = os.getenv("DOCSIGHT_BASE_URL", "https://fastapi.tiangolo.com/")

    answer_model: str = os.getenv("DOCSIGHT_ANSWER_MODEL", "llama-3.3-70b-versatile")
    fast_model: str = os.getenv("DOCSIGHT_FAST_MODEL", "llama-3.1-8b-instant")
    judge_model: str = os.getenv("DOCSIGHT_JUDGE_MODEL", "openai/gpt-oss-120b")
    embed_model: str = os.getenv("DOCSIGHT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = os.getenv("DOCSIGHT_RERANK_MODEL", "BAAI/bge-reranker-base")

    # Retrieval knobs; each is a row in the ablation table
    candidates_per_retriever: int = 30
    fused_top_k: int = 30
    final_top_k: int = 5
    rrf_k: int = 60
    chunk_words: int = 300
    chunk_overlap: int = 40
    # Abstain if the best reranker score is below this (tune on the eval set)
    min_rerank_score: float = float(os.getenv("DOCSIGHT_MIN_RERANK_SCORE", "-2.0"))

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


settings = Settings()
