from docsight.generation.prompts import NOT_FOUND
from docsight.generation.verify import extract_apis, verify
from docsight.llm import FakeLLM
from docsight.pipeline import DocSight
from docsight.retrieval.indexes import BM25Index
from docsight.retrieval.retriever import HybridRetriever

PASSAGES = [
    "[Background Tasks]\nImport BackgroundTasks and call add_task to run work after returning a response.",
    "[CORS]\nAdd CORSMiddleware with allow_origins to let browsers call your API.",
]


def test_extract_apis_ignores_plain_words_and_builtins():
    ans = "Use `BackgroundTasks` and call `add_task`.\n```python\ndef f(x: int):\n    print(x)\n    background_tasks.add_task(send_email)\n```"
    apis = extract_apis(ans)
    assert {"BackgroundTasks", "add_task", "background_tasks.add_task", "send_email"} <= apis
    assert "print" not in apis and "int" not in apis


def test_verify_passes_grounded_answer():
    ans = "Import BackgroundTasks and call add_task to run work after the response [1]."
    v = verify(ans, PASSAGES)
    assert v.passed, v.problems
    assert v.citation_coverage == 1.0 and v.support == 1.0


def test_verify_flags_invented_api_and_bad_citation():
    ans = "Call `app.schedule_task` to run work after returning a response [3]."
    v = verify(ans, PASSAGES)
    assert not v.passed
    assert v.invalid_citations == [3]
    assert "app.schedule_task" in v.unknown_apis


def test_verify_flags_uncited_claims():
    ans = "Background tasks run after the response is sent. They are useful for sending emails to users."
    v = verify(ans, PASSAGES)
    assert v.citation_coverage == 0.0 and not v.passed


def _pipeline(corpus, responder):
    retriever = HybridRetriever(BM25Index(corpus), mode="bm25", final_top_k=3)
    llm = FakeLLM(responder)
    return DocSight(retriever, llm), llm


def test_pipeline_happy_path(corpus):
    ds, llm = _pipeline(corpus, lambda m, model: "Import BackgroundTasks and call add_task to run work after returning a response [1].")
    ans = ds.ask("How do I run work after returning a response with BackgroundTasks?")
    assert not ans.abstained
    assert ans.citations[0].doc_path == "tutorial/background-tasks.md"
    assert len(llm.calls) == 1 and ans.usage["llm_calls"] == 1


def test_pipeline_retries_with_feedback_then_passes(corpus):
    replies = iter([
        "Use `app.run_later()` for that [1].",
        "Import BackgroundTasks and call add_task to run work after returning a response [1].",
    ])
    ds, llm = _pipeline(corpus, lambda m, model: next(replies))
    ans = ds.ask("How do I run work after returning a response with BackgroundTasks?")
    assert not ans.abstained and len(llm.calls) == 2
    assert "failed verification" in llm.calls[1]["messages"][-1]["content"]


def test_pipeline_abstains_when_model_says_not_found(corpus):
    ds, _ = _pipeline(corpus, lambda m, model: NOT_FOUND)
    ans = ds.ask("How do I configure Kubernetes autoscaling for TestClient?")
    assert ans.abstained and ans.citations == []


def test_pipeline_abstains_when_nothing_retrieved(corpus):
    ds, llm = _pipeline(corpus, lambda m, model: "should not be called")
    ans = ds.ask("zzzz qqqq")
    assert ans.abstained and llm.calls == []
