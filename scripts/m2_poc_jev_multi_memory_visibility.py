#!/usr/bin/env python3
"""Bounded current Chinese Jev input visibility probe; no answer or judge calls."""

import hashlib
import json
import time
from pathlib import Path

import requests

from schema_rsi.config import get_settings
from schema_rsi.llm.laya import DEFAULT_LAYA_BASE, LayaClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/analysis/m2_current_zh_stratified_retrieval_20260927.json"
OUT = ROOT / "results/analysis/m2_jev_multi_memory_visibility_20260927.json"
CASE_IDS = ["locomo_conv-42_qa20", "locomo_conv-48_qa107",
            "locomo_conv-26_qa111", "locomo_conv-48_qa2"]
TAIL_CASES = set(CASE_IDS[:2])
ORDER = ("prefix60", "top1_full", "top1_full", "prefix60")
QUESTIONS = {"sufficient": {"type": "noul", "instructions":
             "The context above contains ALL specific facts needed to answer the question completely and precisely (every list item, exact date, and exact value)."}}


def make_input(case, variant):
    memories = case["reranked"][:15]
    assert len(memories) == 15
    contents = [m["content"][:60] for m in memories]
    if variant == "top1_full":
        contents[0] = memories[0]["content"]
    digest = " | ".join(contents)
    assert len(digest) < 950
    return LayaClient.evidence_state(case["question"], digest)


def save(report):
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    settings = get_settings()
    base = ((settings.raw.get("laya") or {}).get("base_url") or DEFAULT_LAYA_BASE).rstrip("/")
    source = json.loads(SOURCE.read_text())
    by_id = {c["case_id"]: c for c in source["cases"]}
    assert set(CASE_IDS) <= by_id.keys()
    report = {"scope": "Current Chinese frozen 15-memory context, Jev decision only; two tail cases and two prefix controls",
              "source": str(SOURCE.relative_to(ROOT)),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "max_decision_calls": 16, "decision_calls": 0,
              "order_per_case": ORDER, "service_identity": "current configured Laya endpoint; checkpoint not archived",
              "cases": [], "stopped_early": False}
    for cid in CASE_IDS:
        case = by_id[cid]
        inputs = {variant: make_input(case, variant) for variant in set(ORDER)}
        assert inputs["prefix60"] != inputs["top1_full"]
        item = {"case_id": cid, "kind": "tail_fact" if cid in TAIL_CASES else "prefix_control",
                "question": case["question"],
                "top1_id": case["reranked"][0]["id"],
                "top1_prefix60": case["reranked"][0]["content"][:60],
                "top1_full": case["reranked"][0]["content"],
                "inputs": {v: {"state_sha256": hashlib.sha256(s.encode()).hexdigest(),
                               "state_chars": len(s)} for v, s in inputs.items()},
                "runs": []}
        report["cases"].append(item)
        for variant in ORDER:
            start = time.monotonic()
            try:
                response = requests.post(base + "/v1/decisions",
                                         json={"state": {"body": inputs[variant][:1000]},
                                               "questions": QUESTIONS}, timeout=15)
                report["decision_calls"] += 1
                event = {"variant": variant, "http_status": response.status_code,
                         "seconds": round(time.monotonic() - start, 3)}
                if response.ok:
                    answer = (response.json().get("answers") or {}).get("sufficient") or {}
                    event["probability"] = answer.get("noul")
                    event["choice"] = answer.get("choice")
                else:
                    event["error_type"] = "non_200"
                    report["stopped_early"] = True
            except requests.RequestException as exc:
                report["decision_calls"] += 1
                event = {"variant": variant, "error_type": type(exc).__name__,
                         "seconds": round(time.monotonic() - start, 3)}
                report["stopped_early"] = True
            item["runs"].append(event)
            save(report)
            if report["stopped_early"]:
                print("stopped after", report["decision_calls"], "request(s);", event.get("error_type"))
                return
    print("completed", report["decision_calls"], "requests")
    for item in report["cases"]:
        print(item["case_id"], [(r["variant"], r.get("probability")) for r in item["runs"]])


if __name__ == "__main__":
    main()
