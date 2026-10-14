"""How far can we trust the LLM judge? Compare its verdicts with yours.

1. Run eval/run_answers.py
2. Open eval/results/judge_review.csv, fill `your_verdict` (correct / partial / incorrect) for ~30 rows
3. python eval/judge_agreement.py
"""

import csv
from collections import Counter

from common import RESULTS


def main() -> None:
    rows = [r for r in csv.DictReader((RESULTS / "judge_review.csv").open()) if r["your_verdict"].strip()]
    if not rows:
        raise SystemExit("No labelled rows yet: fill the your_verdict column first.")
    agree = sum(r["judge_verdict"] == r["your_verdict"].strip().lower() for r in rows)
    print(f"Judge agrees with you on {agree}/{len(rows)} = {agree / len(rows):.0%}")
    confusion = Counter((r["your_verdict"].strip().lower(), r["judge_verdict"]) for r in rows)
    print("you -> judge counts:")
    for (you, j), n in sorted(confusion.items()):
        print(f"  {you:9s} -> {j:9s} {n}")


if __name__ == "__main__":
    main()
