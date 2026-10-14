"""End-to-end answer evaluation (needs GROQ_API_KEY).

    python eval/run_answers.py               # full pipeline, hybrid + rerank
    python eval/run_answers.py --limit 10    # quick check

For every question it records the answer, then asks a judge model (from a
different model family than the answer model, to limit self-preference bias)
to grade it against the gold documentation page.

Metrics:
    answer accuracy     judge says "correct" (answerable questions)
    false abstain rate  abstained on an answerable question
    correct abstain     abstained on an unanswerable question
    verifier pass rate  answers that passed the deterministic checks first time
    p50/p95 latency, tokens per question

It also writes eval/results/judge_review.csv: hand-label 30 rows, then run
eval/judge_agreement.py to report how often the judge agrees with you.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

from common import RESULTS, ROOT, load_questions, markdown_table, mean, percentile, save

from docsight.config import settings
from docsight.llm import GroqLLM
from docsight.pipeline import load_pipeline

JUDGE_PROMPT = """You grade answers from a documentation assistant.

Reference documentation (ground truth):
<<<
{reference}
>>>

Question: {question}

Assistant's answer:
<<<
{answer}
>>>

Grade the answer against the reference only. Reply in JSON:
{{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}
- correct: answers the question and everything it states agrees with the reference
- partial: on the right track but misses the key step or includes a minor error
- incorrect: wrong, misleading, or does not answer the question"""


def reference_text(gold_docs: list[str], max_words: int = 1800) -> str:
    parts = []
    for d in gold_docs:
        p = settings.docs_root / d
        if p.exists():
            parts.append(f"## {d}\n" + " ".join(p.read_text(errors="ignore").split()[:max_words]))
    return "\n\n".join(parts)


def judge(llm: GroqLLM, q: dict, answer: str) -> dict:
    msg = JUDGE_PROMPT.format(reference=reference_text(q["gold_docs"]), question=q["question"], answer=answer)
    resp = llm.complete([{"role": "user", "content": msg}], model=settings.judge_model, json_mode=True, max_tokens=200)
    try:
        out = json.loads(resp.text)
        return {"verdict": out.get("verdict", "incorrect"), "reason": out.get("reason", "")}
    except json.JSONDecodeError:
        return {"verdict": "incorrect", "reason": f"unparseable judge output: {resp.text[:80]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for result files, e.g. 'no-verifier'")
    args = ap.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    llm = GroqLLM(cache_dir=settings.cache_dir / "llm")
    pipe = load_pipeline(llm=llm, use_reranker=not args.no_rerank, mode=args.mode)

    records = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        ans = pipe.ask(q["question"])
        latency = (time.perf_counter() - t0) * 1000
        rec = {"id": q["id"], "question": q["question"], "answerable": q["answerable"], "answer": ans.answer,
               "abstained": ans.abstained, "abstain_reason": ans.abstain_reason, "latency_ms": round(latency, 1),
               "llm_calls": ans.usage.get("llm_calls", 0),
               "tokens": ans.usage.get("prompt_tokens", 0) + ans.usage.get("completion_tokens", 0),
               "first_try_pass": ans.usage.get("llm_calls", 0) == 1 and not ans.abstained,
               "cited_docs": [c.doc_path for c in ans.citations]}
        if q["answerable"] and not ans.abstained:
            rec.update(judge(llm, q, ans.answer))
        records.append(rec)
        print(f"[{i}/{len(questions)}] {q['id']} abstained={ans.abstained} verdict={rec.get('verdict', '-')} {latency:.0f}ms")

    ans_q = [r for r in records if r["answerable"]]
    unans_q = [r for r in records if not r["answerable"]]
    summary = {
        "config": f"{args.mode}{'' if args.no_rerank else ' + rerank'}{(' / ' + args.tag) if args.tag else ''}",
        "n": len(records),
        "accuracy": round(mean([1.0 if r.get("verdict") == "correct" else 0.0 for r in ans_q]), 3),
        "accuracy_incl_partial": round(mean([1.0 if r.get("verdict") in ("correct", "partial") else 0.0 for r in ans_q]), 3),
        "false_abstain_rate": round(mean([1.0 if r["abstained"] else 0.0 for r in ans_q]), 3),
        "correct_abstain_rate": round(mean([1.0 if r["abstained"] else 0.0 for r in unans_q]), 3) if unans_q else None,
        "first_try_verifier_pass": round(mean([1.0 if r["first_try_pass"] else 0.0 for r in records]), 3),
        "p50_ms": round(percentile([r["latency_ms"] for r in records], 0.5)),
        "p95_ms": round(percentile([r["latency_ms"] for r in records], 0.95)),
        "tokens_per_q": round(mean([r["tokens"] for r in records])),
    }
    print(json.dumps(summary, indent=2))
    name = "answers" + (f"_{args.tag}" if args.tag else "")
    table = markdown_table([summary], [(k, k) for k in summary])
    save(name, {"summary": summary, "records": records}, table)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "judge_review.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "judge_verdict", "judge_reason", "your_verdict"])
        for r in records:
            if "verdict" in r:
                w.writerow([r["id"], r["question"], r["answer"], r["verdict"], r["reason"], ""])
    print(f"Wrote {RESULTS / 'judge_review.csv'}: fill in your_verdict for ~30 rows, then run eval/judge_agreement.py")


if __name__ == "__main__":
    main()
