# Development timeline

DocSight was developed locally between **July 15 and October 14, 2026**, without regular Git commits along the way. This page breaks that work into 34 milestones in the order they were built.

> **About the dates.** The dates below are **estimates based on my recollection** of the development period. They are not Git timestamps. Every commit in this repository carries its real timestamp (October 14, 2026, when the code was first committed). The dates here have not been changed to match this table.

## How the commit history was built

The original history is 6 large commits on `main`. The `reconstructed-history` branch splits them into the 34 commits below:

- Each commit adds or updates whole files, copied unchanged from the original commit that introduced them. No intermediate code was written; files are not split internally.
- Tests come right after the code they cover, and the full test suite passes at every commit.
- The final code on `reconstructed-history` is identical to `main`.

## Milestones

| # | Estimated date | Commit | Message | Original commit |
|---|---|---|---|---|
| 1 | Jul 15 | `bef1880` | chore: scaffold package layout, dependencies and gitignore | `619ad12` |
| 2 | Jul 17 | `413aa1f` | feat(config): add env-driven settings and .env example | `619ad12` |
| 3 | Jul 20 | `bc8581a` | feat(models): add Chunk, ScoredChunk, Citation and Answer models | `619ad12` |
| 4 | Jul 24 | `fbcd069` | feat(ingest): parse markdown into heading-aware sections with includes | `619ad12` |
| 5 | Jul 29 | `3024194` | feat(ingest): chunk sections without splitting code blocks | `619ad12` |
| 6 | Aug 1 | `ede1b1b` | test(ingest): cover section parsing, page URLs and chunking | `619ad12` |
| 7 | Aug 5 | `3390944` | feat(retrieval): add FastEmbed and hash-based embedders | `619ad12` |
| 8 | Aug 9 | `1e3b8a1` | feat(retrieval): add reciprocal rank fusion | `619ad12` |
| 9 | Aug 12 | `e61931f` | feat(retrieval): add cross-encoder reranker interface | `619ad12` |
| 10 | Aug 16 | `dc171fd` | feat(retrieval): add BM25 and Qdrant dense indexes with chunk persistence | `6e42d43` |
| 11 | Aug 21 | `1700af9` | feat(retrieval): combine sparse and dense search in HybridRetriever | `6e42d43` |
| 12 | Aug 24 | `0963486` | test(retrieval): cover tokenizer, indexes, RRF and retriever modes | `6e42d43` |
| 13 | Aug 28 | `b453891` | feat(llm): add Groq client with retries and on-disk cache | `144d7b1` |
| 14 | Aug 31 | `fb69419` | feat(tracing): add optional Langfuse spans | `144d7b1` |
| 15 | Sep 3 | `a50be63` | feat(generation): add citation-numbered answer prompts | `144d7b1` |
| 16 | Sep 8 | `d87250e` | feat(generation): verify citations, support and API grounding | `144d7b1` |
| 17 | Sep 10 | `e81d403` | refactor(retrieval): mark rerankers as score-calibrated or not | `144d7b1` |
| 18 | Sep 14 | `065dfbf` | feat(pipeline): wire retrieval, generation and verify-retry loop | `144d7b1` |
| 19 | Sep 16 | `c72c767` | test(generation): cover verifier and pipeline abstain/retry paths | `144d7b1` |
| 20 | Sep 19 | `d143164` | chore(scripts): add FastAPI docs fetch script | `2243157` |
| 21 | Sep 21 | `7c9722a` | feat(scripts): add index build script | `2243157` |
| 22 | Sep 24 | `d075b45` | feat(api): add FastAPI service with /ask, /retrieve and /health | `2243157` |
| 23 | Sep 25 | `5ffcf8a` | test(api): cover ask/retrieve endpoints and validation | `2243157` |
| 24 | Sep 28 | `3e835ff` | feat(ui): add Streamlit question-answering UI | `2243157` |
| 25 | Sep 30 | `8ac366c` | chore(eval): add shared eval helpers and ignore results | `51217a8` |
| 26 | Oct 2 | `b8aa6b4` | feat(eval): add 50-question evaluation set | `51217a8` |
| 27 | Oct 4 | `3ee35b2` | feat(eval): add retrieval ablation with Hit@k, MRR and latency | `51217a8` |
| 28 | Oct 7 | `1ffc2d4` | feat(eval): add LLM-judged answer evaluation | `51217a8` |
| 29 | Oct 8 | `4949ede` | feat(eval): add judge agreement check | `51217a8` |
| 30 | Oct 10 | `010e6e5` | chore: add pyproject with pytest config | `dee9e63` |
| 31 | Oct 11 | `5339f56` | chore(ci): run unit tests and offline retrieval smoke test | `dee9e63` |
| 32 | Oct 12 | `f34162b` | chore(docker): add Dockerfile and compose file | `dee9e63` |
| 33 | Oct 13 | `12cd867` | chore: add Makefile for common tasks | `dee9e63` |
| 34 | Oct 14 | `c10de21` | docs: add README | `dee9e63` |

## Phases

| Phase | Estimated period | Milestones |
|---|---|---|
| Setup, config and data models | Jul 15 – Jul 20 | 1–3 |
| Ingestion and chunking | Jul 24 – Aug 1 | 4–6 |
| Embeddings, fusion, reranking, indexing, hybrid retrieval | Aug 5 – Aug 24 | 7–12 |
| LLM integration, verification, RAG pipeline | Aug 28 – Sep 16 | 13–19 |
| Index build, API and UI | Sep 19 – Sep 28 | 20–24 |
| Evaluation | Sep 30 – Oct 8 | 25–29 |
| Packaging, CI, Docker and docs | Oct 10 – Oct 14 | 30–34 |
