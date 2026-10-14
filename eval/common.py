"""Shared helpers for the eval scripts."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "eval" / "results"


def load_questions(path: Path = ROOT / "eval" / "questions.jsonl") -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(r.get(key, "")) for key, _ in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def save(name: str, payload: dict | list, table: str | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2))
    if table:
        (RESULTS / f"{name}.md").write_text(table + "\n")
