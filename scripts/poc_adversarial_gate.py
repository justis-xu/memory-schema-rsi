#!/usr/bin/env python3
"""Eight-case capped audit of Chinese adversarial lenient-gate failures.

This is a deliberately small, fixed sample. It does not alter original grades.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation.precision_judge import PrecisionJudge  # noqa: E402

CASE_IDS = (
    "locomo_conv-26_qa179",  # refusal plus grounded adjacent fact
    "locomo_conv-30_qa103",  # wrong subject correction
    "locomo_conv-41_qa169",  # wrong subject correction
    "locomo_conv-42_qa222",  # clean refusal
    "locomo_conv-43_qa182",  # wrong subject correction
    "locomo_conv-44_qa134",  # wrong subject correction
    "locomo_conv-47_qa171",  # refusal followed by a likely unsupported answer
    "locomo_conv-49_qa190",  # clean refusal plus adjacent facts
)
OUTPUT = ROOT / "results/analysis/adversarial_gate_poc_20260926.json"
CACHE = ROOT / "results/analysis/adversarial_gate_poc_cache.json"


def main() -> None:
    rows = {
        row["case_id"]: row
        for line in (ROOT / "results/precision_regrade_zhfull.jsonl").read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    assert len(CASE_IDS) == 8
    for case_id in CASE_IDS:
        row = rows[case_id]
        assert row["category"] == "adversarial" and row["final"] == "wrong" and row["judge"] is None
    settings = get_settings("config/locomo_zh.yaml")
    judge = PrecisionJudge(settings=settings, model="glm-5.3", cache_path=str(CACHE))
    results = []
    for case_id in CASE_IDS:
        row = rows[case_id]
        started = time.monotonic()
        try:
            verdict = judge.grade(row["category"], row["question"], row["gold"], row["pred"])
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            verdict = {"error": str(exc)[:200]}
            status = "error"
        results.append({
            "case_id": case_id, "question": row["question"], "archived_grade": row["final"],
            "archived_lenient": row["lenient"], "answer": row["pred"],
            "strict_verdict": verdict, "status": status,
            "duration_s": round(time.monotonic() - started, 2),
        })
        print(case_id, status, verdict.get("grade"), flush=True)
        if status == "error":
            break  # preserve the call budget when the service is failing
    judge.flush()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "selection": "Eight named Chinese adversarial cases archived as wrong solely because lenient=false; seven contain a refusal/correction, one includes a likely guessed answer. Fixed before new judging.",
        "max_cases": 8, "model": "glm-5.3", "results": results,
        "limits": "Purposeful sample, not an estimate of population error rate. The judge itself may be inconsistent; inspect source evidence before labeling hallucination.",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
