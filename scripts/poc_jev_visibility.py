#!/usr/bin/env python3
"""Twelve-call, fixed-case probe of Jev score sensitivity to hidden tail facts.

This deliberately uses one memory per case, so it isolates visibility rather
than recreating the archived 15-memory context or its historical service.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl"
OUTPUT = ROOT / "results/analysis/jev_visibility_poc_20260926.json"
CASE_IDS = (
    "locomo_conv-41_qa111",
    "locomo_conv-42_qa98",
    "locomo_conv-42_qa102",
)
INSTRUCTIONS = (
    "The context above contains ALL specific facts needed to answer the "
    "question completely and precisely (every list item, exact date, and exact value)."
)


def cases() -> list[dict]:
    selected = {}
    with SOURCE.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["case_id"] in CASE_IDS:
                selected[row["case_id"]] = row
    assert set(selected) == set(CASE_IDS), set(CASE_IDS) - set(selected)
    result = []
    for case_id in CASE_IDS:
        row = selected[case_id]
        gold = row["expected_answer"]
        memories = row["retrieved_memories"] + row.get("graph_memories", [])
        matches = [mem for mem in memories if gold in mem["content"] and gold not in mem["content"][:60]]
        assert matches, case_id
        mem = matches[0]
        result.append({
            "case_id": case_id, "question": row["question"], "gold": gold,
            "memory_id": mem["id"], "short": mem["content"][:60], "full": mem["content"],
        })
    return result


def score(base: str, question: str, context: str) -> dict:
    body = f"Question: {question}\n\nContext:\n{context[:950]}"
    payload = {
        "state": {"body": body[:1000]},
        "questions": {"sufficient": {"type": "noul", "instructions": INSTRUCTIONS}},
    }
    started = time.monotonic()
    try:
        response = requests.post(f"{base.rstrip('/')}/v1/decisions", json=payload, timeout=15)
        result = {"http_status": response.status_code, "seconds": round(time.monotonic() - started, 3)}
        if response.ok:
            answer = (response.json().get("answers") or {}).get("sufficient") or {}
            result["noul"] = answer.get("noul")
            result["choice"] = answer.get("choice")
        else:
            result["error"] = response.text[:160]
        return result
    except requests.RequestException as exc:
        return {"error": str(exc)[:160], "seconds": round(time.monotonic() - started, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laya-url", required=True)
    parser.add_argument("--decider-url", required=True)
    args = parser.parse_args()
    report = {
        "purpose": "Single-memory visibility sensitivity; not a replication of historical 15-memory Jev decisions.",
        "max_calls": 12, "cases": [],
    }
    for case in cases():
        record = {**case, "scores": {}}
        for service, base in (("laya", args.laya_url), ("decider", args.decider_url)):
            record["scores"][service] = {
                variant: score(base, case["question"], case[variant])
                for variant in ("short", "full")
            }
        report["cases"].append(record)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT), "calls": sum(len(service) for case in report["cases"] for service in case["scores"].values()),
        "scores": [{"case_id": c["case_id"], "scores": c["scores"]} for c in report["cases"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
