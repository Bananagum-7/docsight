import pytest

from docsight.models import ScoredChunk
from docsight.retrieval.embed import HashEmbedder
from docsight.retrieval.fusion import rrf
from docsight.retrieval.indexes import BM25Index, DenseIndex, load_chunks, save_chunks, tokenize
from docsight.retrieval.retriever import HybridRetriever


def test_tokenize_keeps_identifiers():
    assert tokenize("How do I use add_task with app.state?") == ["use", "add_task", "app.state"]


def test_bm25_finds_exact_api_name(corpus):
    hits = BM25Index(corpus).search("UploadFile", k=3)
    assert hits[0].chunk.id == "c5"


def test_dense_index_roundtrip(corpus):
    dense = DenseIndex(HashEmbedder())
    dense.build(corpus)
    hits = dense.search("CORSMiddleware allow_origins browsers", k=2)
    assert hits[0].chunk.id == "c2"


def test_rrf_rewards_agreement(corpus):
    a = [ScoredChunk(chunk=corpus[0], score=9), ScoredChunk(chunk=corpus[1], score=5)]
    b = [ScoredChunk(chunk=corpus[1], score=0.9), ScoredChunk(chunk=corpus[2], score=0.8)]
    fused = rrf([a, b], k=60)
    # c2 is ranked by both retrievers, so it wins even though it is top in neither
    assert fused[0].chunk.id == "c2"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_retriever_modes(corpus):
    dense = DenseIndex(HashEmbedder())
    dense.build(corpus)
    for mode in ("bm25", "dense", "hybrid"):
        res = HybridRetriever(BM25Index(corpus), dense, mode=mode, final_top_k=3).retrieve("TestClient pytest")
        assert res.chunks[0].chunk.id == "c4", mode
        assert set(res.timings_ms) == {"search", "fuse", "rerank"}


def test_dense_mode_requires_index(corpus):
    with pytest.raises(ValueError):
        HybridRetriever(BM25Index(corpus), None, mode="hybrid")


def test_chunk_persistence(tmp_path, corpus):
    path = tmp_path / "chunks.jsonl"
    save_chunks(corpus, path)
    assert load_chunks(path) == corpus
