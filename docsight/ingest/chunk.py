"""Split sections into retrievable chunks without breaking code blocks.

A section that fits in `size` words becomes one chunk. A longer section is
split into units (paragraphs and whole code blocks), which are packed into
chunks of up to `size` words, carrying the last `overlap` words of prose into
the next chunk so a sentence cut at the boundary is still retrievable.

Every chunk starts with a header like
    [Background Tasks > Using BackgroundTasks]
so retrieval sees where in the docs the text lives.
"""

from __future__ import annotations

import hashlib

from docsight.ingest.markdown import FENCE_RE, Section, page_url
from docsight.models import Chunk


def _id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def split_units(text: str) -> list[str]:
    """Paragraphs and fenced code blocks, each kept whole."""
    units, buf, in_fence = [], [], False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            if not in_fence and buf:
                units.append("\n".join(buf).strip())
                buf = []
            buf.append(line)
            in_fence = not in_fence
            if not in_fence:
                units.append("\n".join(buf).strip())
                buf = []
            continue
        if not in_fence and not line.strip():
            if buf:
                units.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        units.append("\n".join(buf).strip())
    return [u for u in units if u]


def pack(units: list[str], size: int, overlap: int) -> list[str]:
    chunks, cur, count = [], [], 0
    for u in units:
        n = len(u.split())
        if cur and count + n > size:
            chunks.append("\n\n".join(cur))
            tail = [] if cur[-1].lstrip().startswith("```") else cur[-1].split()[-overlap:]
            cur = [" ".join(tail)] if tail else []
            count = len(tail)
        cur.append(u)
        count += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_sections(sections: list[Section], base_url: str, size: int = 300, overlap: int = 40) -> list[Chunk]:
    out: list[Chunk] = []
    for s_idx, s in enumerate(sections):
        pieces = pack(split_units(s.text), size, overlap)
        for p_idx, piece in enumerate(pieces):
            out.append(
                Chunk(
                    id=_id(s.doc_path, s.anchor, s_idx, p_idx),
                    text=f"[{s.breadcrumb}]\n{piece}",
                    doc_path=s.doc_path,
                    title=s.title,
                    breadcrumb=s.breadcrumb,
                    url=page_url(base_url, s.doc_path, s.anchor),
                    has_code="```" in piece,
                )
            )
    return out
