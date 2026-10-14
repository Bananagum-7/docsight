"""Build the search index: parse docs -> chunk -> BM25 corpus + Qdrant vectors.

    python scripts/build_index.py                # real embeddings (bge-small, CPU)
    python scripts/build_index.py --embedder hash  # no model download (offline / CI)

Writes data/index/chunks.jsonl, data/index/qdrant/ and data/index/manifest.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsight.config import settings  # noqa: E402
from docsight.ingest.chunk import chunk_sections  # noqa: E402
from docsight.ingest.markdown import iter_markdown_files, parse_markdown  # noqa: E402
from docsight.retrieval.indexes import DenseIndex, save_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-root", type=Path, default=settings.docs_root)
    ap.add_argument("--code-root", type=Path, default=settings.code_root)
    ap.add_argument("--embedder", choices=["fastembed", "hash"], default="fastembed")
    ap.add_argument("--no-dense", action="store_true", help="only build the BM25 corpus")
    args = ap.parse_args()

    if not args.docs_root.exists():
        sys.exit(f"Docs not found at {args.docs_root}. Run: bash scripts/fetch_docs.sh")

    t0 = time.time()
    sections = []
    files = list(iter_markdown_files(args.docs_root))
    for f in files:
        sections.extend(parse_markdown(f, args.docs_root, args.code_root))
    chunks = chunk_sections(sections, settings.base_url, settings.chunk_words, settings.chunk_overlap)
    out = settings.index_dir
    save_chunks(chunks, out / "chunks.jsonl")
    print(f"Parsed {len(files)} files -> {len(sections)} sections -> {len(chunks)} chunks ({time.time() - t0:.1f}s)")

    if not args.no_dense:
        if args.embedder == "hash":
            from docsight.retrieval.embed import HashEmbedder

            embedder = HashEmbedder()
        else:
            from docsight.retrieval.embed import FastEmbedEmbedder

            embedder = FastEmbedEmbedder(settings.embed_model)
        t1 = time.time()
        DenseIndex(embedder, path=out / "qdrant").build(chunks)
        print(f"Embedded {len(chunks)} chunks with {args.embedder} in {time.time() - t1:.1f}s")

    (out / "manifest.json").write_text(json.dumps({
        "files": len(files), "sections": len(sections), "chunks": len(chunks),
        "embedder": None if args.no_dense else (args.embedder if args.embedder == "hash" else settings.embed_model),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))


if __name__ == "__main__":
    main()
