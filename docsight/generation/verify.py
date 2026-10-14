"""Deterministic checks on a generated answer, run before it reaches the user.

1. Citation validity  - every [n] refers to a passage that exists.
2. Citation coverage  - share of factual sentences that carry a citation.
3. Support            - share of cited sentences whose content words mostly
                        appear in the passages they cite (a cheap, fast
                        faithfulness proxy; the eval uses an LLM judge too).
4. API grounding      - every API-looking identifier in the answer (code
                        spans and blocks) appears somewhere in the retrieved
                        passages. This catches the classic docs-bot failure:
                        confidently inventing a function that doesn't exist.

None of these call an LLM, so they add ~1 ms and cost nothing.
"""

from __future__ import annotations

import builtins
import keyword
import re
from dataclasses import asdict, dataclass, field

from docsight.retrieval.indexes import tokenize

CITE_RE = re.compile(r"\[(\d+)\]")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\[])")

IGNORE_IDENTS = set(dir(builtins)) | set(keyword.kwlist) | {
    "self", "cls", "app", "item", "items", "user", "users", "db", "q", "x", "y", "data", "result", "response", "request",
    "main", "name", "value", "id", "async", "await", "def", "return", "import", "from", "true", "false", "null",
}


@dataclass
class Verification:
    invalid_citations: list[int] = field(default_factory=list)
    citation_coverage: float = 1.0
    support: float = 1.0
    unknown_apis: list[str] = field(default_factory=list)
    passed: bool = True
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _api_like(ident: str) -> bool:
    """CamelCase, dotted.access or snake_case names look like APIs; plain words don't."""
    base = ident.split(".")[-1]
    return ("." in ident) or ("_" in base.strip("_")) or (any(ch.isupper() for ch in base[1:]) and base[0].isupper())


def extract_apis(answer: str) -> set[str]:
    code = " ".join(CODE_BLOCK_RE.findall(answer)) + " " + " ".join(INLINE_CODE_RE.findall(answer))
    found = set()
    for ident in IDENT_RE.findall(code):
        if ident in IGNORE_IDENTS:
            continue
        if _api_like(ident):
            found.add(ident)
    return found


def prose_sentences(answer: str) -> list[str]:
    prose = CODE_BLOCK_RE.sub(" ", answer)
    sentences = []
    for line in prose.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if len(line.split()) < 4:  # headings, list labels, fragments
            continue
        sentences.extend(s for s in SENT_RE.split(line) if len(s.split()) >= 4)
    return sentences


def verify(answer: str, passages: list[str], min_coverage: float = 0.6, min_support: float = 0.6, support_overlap: float = 0.5) -> Verification:
    v = Verification()
    n = len(passages)

    cited = [int(m) for m in CITE_RE.findall(answer)]
    v.invalid_citations = sorted({c for c in cited if c < 1 or c > n})

    sentences = prose_sentences(answer)
    with_cites = [s for s in sentences if CITE_RE.search(s)]
    v.citation_coverage = len(with_cites) / len(sentences) if sentences else 1.0

    supported = 0
    for s in with_cites:
        words = set(tokenize(CITE_RE.sub("", s)))
        refs = [int(m) for m in CITE_RE.findall(s) if 1 <= int(m) <= n]
        source = set(tokenize(" ".join(passages[r - 1] for r in refs)))
        if words and len(words & source) / len(words) >= support_overlap:
            supported += 1
    v.support = supported / len(with_cites) if with_cites else (1.0 if not sentences else 0.0)

    corpus = "\n".join(passages)
    v.unknown_apis = sorted(a for a in extract_apis(answer) if a not in corpus and a.split(".")[-1] not in corpus)

    if v.invalid_citations:
        v.problems.append(f"citations {v.invalid_citations} do not match any passage")
    if v.citation_coverage < min_coverage:
        v.problems.append(f"only {v.citation_coverage:.0%} of sentences are cited")
    if v.support < min_support:
        v.problems.append(f"only {v.support:.0%} of cited sentences are supported by their passages")
    if v.unknown_apis:
        v.problems.append(f"uses names not found in the docs: {', '.join(v.unknown_apis)}")
    v.passed = not v.problems
    return v
