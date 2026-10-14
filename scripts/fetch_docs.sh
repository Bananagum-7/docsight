#!/usr/bin/env bash
# Download the FastAPI docs (MIT licensed) and their code examples into data/fastapi.
# Only the folders we need are checked out, so this takes a few seconds.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/fastapi
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull --depth 1 -q
else
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git "$DEST"
  git -C "$DEST" sparse-checkout set docs/en/docs docs_src
fi
echo "Docs at $DEST/docs/en/docs ($(find $DEST/docs/en/docs -name '*.md' | wc -l) markdown files)"
