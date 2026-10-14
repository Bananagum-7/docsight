from fastapi.testclient import TestClient

from docsight import api
from docsight.llm import FakeLLM
from docsight.pipeline import DocSight
from docsight.retrieval.indexes import BM25Index
from docsight.retrieval.retriever import HybridRetriever


def make_client(corpus):
    retriever = HybridRetriever(BM25Index(corpus), mode="bm25", final_top_k=3)
    llm = FakeLLM(lambda m, model: "Add CORSMiddleware with allow_origins to let browsers call your API [1].")
    api.state["pipeline"] = DocSight(retriever, llm)
    return TestClient(api.app)


def test_ask_and_retrieve(corpus):
    with make_client(corpus) as client:
        assert client.get("/health").json() == {"status": "ok", "ready": True}
        r = client.post("/ask", json={"question": "How do I enable CORSMiddleware for browsers?"})
        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is False and body["citations"][0]["doc_path"] == "tutorial/cors.md"
        r = client.post("/retrieve", json={"question": "UploadFile", "top_k": 2})
        assert r.json()["results"][0]["breadcrumb"] == "Request Files > UploadFile"
    api.state.clear()


def test_validation_error(corpus):
    with make_client(corpus) as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
    api.state.clear()
