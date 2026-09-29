"""Count archived answer exposure to one source-misdated Chinese memory.

This is a string-level archive audit, not an effect or causal estimate.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/analysis/m2_output_origin_sample_20260928.json"
OUT = ROOT / "results/analysis/m2_misdated_memory_answer_exposure_20260929.json"
MEMORY_ID = "f87df58f-81e4-46e4-a2bc-22e62b98c625"
DATE_RE = re.compile(r"10月20日|2023-10-20|十月二十日")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads(SOURCE.read_text())
    expected_ids = set(source["mismatched_output_answer_context_exposure"][MEMORY_ID]["unique_case_ids"])
    assert len(expected_ids) == 28
    runs = sorted((ROOT / "results_zh").glob("zhfull*jsonl"))
    assert len(runs) == 8, "Archive set changed; review before recounting"
    per_case = defaultdict(list)
    run_hashes = {}
    for run in runs:
        run_hashes[run.name] = sha256(run)
        for raw in run.open():
            row = json.loads(raw)
            if MEMORY_ID not in row.get("metadata", {}).get("answer_context_ids", []):
                continue
            case_id = row["case_id"]
            assert case_id in expected_ids
            answer = row.get("predicted_answer") or ""
            per_case[case_id].append({
                "run": run.name,
                "predicted_answer": answer,
                "mentions_oct20": bool(DATE_RE.search(answer)),
                "metrics": row.get("metrics"),
            })
            if len(per_case[case_id]) == 1:
                per_case[case_id + ":question"] = row["question"]
                per_case[case_id + ":gold"] = row.get("expected_answer")
            else:
                assert per_case[case_id + ":question"] == row["question"]
                assert per_case[case_id + ":gold"] == row.get("expected_answer")
    assert set(k for k in per_case if ":" not in k) == expected_ids
    rows = []
    for case_id in sorted(expected_ids, key=lambda x: int(x.rsplit("qa", 1)[1])):
        answers = per_case[case_id]
        rows.append({
            "case_id": case_id,
            "question": per_case[case_id + ":question"],
            "expected_answer": per_case[case_id + ":gold"],
            "exposure_count": len(answers),
            "oct20_mention_count": sum(x["mentions_oct20"] for x in answers),
            "answer_instances": answers,
        })
    total = sum(r["exposure_count"] for r in rows)
    mentions = sum(r["oct20_mention_count"] for r in rows)
    assert total == 219, total
    data = {
        "scope": "Eight archived zhfull runs; rows whose final answer_context_ids include the fixed source-misdated memory. Literal date mentions are not causal attribution or semantic error labels.",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "memory_id": MEMORY_ID,
        "archived_run_sha256": run_hashes,
        "counts": {"runs": len(runs), "unique_questions": len(rows), "answer_exposures": total,
                   "answers_mentioning_oct20": mentions,
                   "questions_with_oct20_mention": sum(r["oct20_mention_count"] > 0 for r in rows)},
        "questions": rows,
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(data["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
