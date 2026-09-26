#!/usr/bin/env python3
"""Attach manually reviewed fact labels to the fixed 12-call graph-swap POC."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/analysis/m2_frozen_graph_swaps_poc_20260926.json"
OUT = ROOT / "results/analysis/m2_frozen_graph_swaps_labels_20260926.json"

# Each boolean is a manual judgment of the actual final answer against the
# specified source-backed facts, not an old strict-judge label.
LABELS = {
    "locomo_conv-47_qa96": [False, True, True, False],
    "locomo_conv-41_qa86": [True, False, False, True],
    "locomo_conv-30_qa80": [True, True, True, True],
}
CRITERIA = {
    "locomo_conv-47_qa96": "最终答案明确说约翰支持曼彻斯特城；不能把詹姆斯的利物浦归给约翰",
    "locomo_conv-41_qa86": "最终答案指出约翰的车抛锚且维修费让钱包吃不消；慈善开销没有源话支持",
    "locomo_conv-30_qa80": "最终答案包含完善商业计划、调整投资推介、搭建线上平台三项；额外社媒泛谈不替代必需项",
}


def final_answer(text):
    parts = re.split(r"\*{0,2}ANSWER\*{0,2}\s*[:：]", text, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def main():
    raw = json.loads(RAW.read_text())
    assert sum(len(case["calls"]) for case in raw["cases"]) == 12
    rows = []
    for case in raw["cases"]:
        labels = LABELS[case["case_id"]]
        assert [call["arm"] for call in case["calls"]] == ["base", "swap", "swap", "base"]
        calls = []
        for call, correct in zip(case["calls"], labels):
            assert "answer" in call
            calls.append({
                "index": call["index"], "arm": call["arm"],
                "final_answer": final_answer(call["answer"]),
                "contains_source_supported_required_fact": correct,
            })
        rows.append({"case_id": case["case_id"], "criterion": CRITERIA[case["case_id"]], "calls": calls})
    usage = {
        "calls": 12,
        "prompt_tokens": sum(call["usage"].get("prompt_tokens", 0) for case in raw["cases"] for call in case["calls"]),
        "completion_tokens": sum(call["usage"].get("completion_tokens", 0) for case in raw["cases"] for call in case["calls"]),
    }
    OUT.write_text(json.dumps({
        "scope": "Manual fact-level labels for 12 fixed-context generation calls",
        "raw_results": str(RAW.relative_to(ROOT)), "usage": usage, "cases": rows,
        "limits": ["Retrospectively selected three cases; no population effect size", "Two samples per arm; no judge calls", "Only one memory slot changed; historical multi-slot graph path not reproduced"],
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}: {usage}")


if __name__ == "__main__":
    main()
