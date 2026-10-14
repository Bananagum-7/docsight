"""Parse a docs site (a folder of markdown files) into heading-scoped sections.

Why heading-aware parsing instead of fixed-size chunks:
- A heading is the author's own boundary for "one idea". Chunks that follow
  headings answer one question each, which is what retrieval needs.
- The heading path ("Security > OAuth2 with JWT > Hash the password") is
  prepended to every chunk. It gives both BM25 and the embedding model
  context that the paragraph itself often lacks.
- Code blocks are never split. Half a code example is worse than none.

Docs that include code from other files (FastAPI uses `{* ../../docs_src/x.py *}`)
get the file inlined, so example code is searchable too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*(?:\{\s*#([\w\-]+)\s*\})?\s*$")
INCLUDE_RE = re.compile(r"^\{\*\s*(\S+?\.py)(?:\s+[^*]*)?\*\}\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SKIP_FILES = {"release-notes.md", "fastapi-people.md", "external-links.md", "newsletter.md", "translation-banner.md", "_llm-test.md"}


@dataclass
class Section:
    doc_path: str
    title: str
    headings: list[str]
    anchor: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.headings)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def clean_heading(text: str) -> str:
    return re.sub(r"[`*_]", "", text).strip()


def resolve_include(target: str, md_file: Path, code_root: Path | None) -> str | None:
    """Turn `{* ../../docs_src/a/b.py hl[1] *}` into a fenced python block."""
    candidates = [(md_file.parent / target).resolve()]
    if code_root is not None and "docs_src/" in target:
        candidates.append(code_root / ("docs_src/" + target.split("docs_src/", 1)[1]))
    for path in candidates:
        if path.is_file():
            code = path.read_text(encoding="utf-8", errors="ignore").strip()
            return f"```python\n{code}\n```"
    return None


def parse_markdown(md_file: Path, docs_root: Path, code_root: Path | None = None) -> list[Section]:
    doc_path = md_file.relative_to(docs_root).as_posix()
    raw = md_file.read_text(encoding="utf-8", errors="ignore").splitlines()

    sections: list[Section] = []
    stack: list[str] = []
    title = Path(doc_path).stem.replace("-", " ").title()
    current = Section(doc_path, title, [title], "")
    in_fence = False

    for line in raw:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            current.lines.append(line)
            continue
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            level, text, anchor = len(m.group(1)), clean_heading(m.group(2)), m.group(3)
            if current.text:
                sections.append(current)
            if level == 1:
                title, stack = text, [text]
            else:
                stack = stack[: level - 1] + [text]
            current = Section(doc_path, title, list(stack), anchor or slugify(text))
            continue
        inc = None if in_fence else INCLUDE_RE.match(line.strip())
        if inc:
            code = resolve_include(inc.group(1), md_file, code_root)
            if code:
                current.lines.append(code)
            continue
        current.lines.append(line)

    if current.text:
        sections.append(current)
    for s in sections:  # every section knows its page title, even the ones before the H1
        s.title = title
    return sections


def iter_markdown_files(docs_root: Path):
    for path in sorted(docs_root.rglob("*.md")):
        if path.name in SKIP_FILES:
            continue
        yield path


def page_url(base_url: str, doc_path: str, anchor: str) -> str:
    slug = doc_path[:-3]
    slug = "" if slug == "index" else slug.removesuffix("/index").removesuffix("index") + "/"
    return f"{base_url.rstrip('/')}/{slug}" + (f"#{anchor}" if anchor else "")
