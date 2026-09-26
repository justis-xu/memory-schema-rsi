#!/usr/bin/env python3
"""在中文归档配对运行中，寻找答案几乎同文但 strict exact 翻转的题。"""

from __future__ import annotations

import difflib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import audit_history as history

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/analysis/m2_precision_flip_20260926.json"
ARMS = (
    "zh_baseline_repeat", "zh_graph_repeat", "zh_graph", "zh_jevdec", "zh_jevlaya"
)


def normalized(text: str) -> str:
    """仅用于筛选：NFKC、大小写及标点/空白归一；不代表语义等价。"""
    return "".join(
        ch.lower() for ch in unicodedata.normalize("NFKC", text) if ch.isalnum()
    )


def audit_arm(name: str) -> dict:
    left_grades, right_grades = (history.read_jsonl(path) for path in history.ARMS[name])
    left_raw, right_raw = (history.read_jsonl(path) for path in history.RAW[name])
    counts = Counter()
    close_flips = []
    for case_id in sorted(set(left_grades) & set(right_grades) & set(left_raw) & set(right_raw)):
        left_grade, right_grade = left_grades[case_id], right_grades[case_id]
        if left_grade["category"] not in history.J_CATS:
            continue
        assert left_grade["gold"] == right_grade["gold"], case_id
        left, right = left_raw[case_id], right_raw[case_id]
        answer_a, answer_b = left["predicted_answer"], right["predicted_answer"]
        normalized_a, normalized_b = normalized(answer_a), normalized(answer_b)
        ratio = difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()
        equal_after_normalization = normalized_a == normalized_b
        same_context = history.context_fingerprint(left) == history.context_fingerprint(right)
        flipped = (left_grade["final"] == "exact") != (right_grade["final"] == "exact")
        counts["paired_J"] += 1
        counts["same_full_context"] += same_context
        counts["exact_flips"] += flipped
        counts["same_context_exact_flips"] += same_context and flipped
        counts["normalized_equal_answers"] += equal_after_normalization
        counts["normalized_equal_exact_flips"] += equal_after_normalization and flipped
        counts["near_same_answer_exact_flips"] += ratio >= 0.9 and flipped
        counts["near_same_answer_same_context_exact_flips"] += ratio >= 0.9 and same_context and flipped
        if ratio >= 0.9 and same_context and flipped:
            close_flips.append({
                "case_id": case_id,
                "category": left_grade["category"],
                "similarity": round(ratio, 4),
                "normalized_equal": equal_after_normalization,
                "gold": left_grade["gold"],
                "left_answer": answer_a,
                "right_answer": answer_b,
                "left_final": left_grade["final"],
                "right_final": right_grade["final"],
                "left_lenient": left_grade["lenient"],
                "right_lenient": right_grade["lenient"],
                "left_judge": left_grade["judge"],
                "right_judge": right_grade["judge"],
            })
    return {"counts": dict(counts), "near_same_answer_same_context_flips": close_flips}


def main() -> None:
    arms = {name: audit_arm(name) for name in ARMS}
    report = {
        "scope": "中文 2026-09-25 严格重判的 5 组配对 J 类运行",
        "method": "NFKC 归一化并去标点/空白后计算 SequenceMatcher 字符相似度；记录相似度≥0.9、最终上下文完全相同、exact 翻转的逐题答案与裁判理由。",
        "limits": "0.9 是人工核查筛选阈值，不是语义等价判定；高度相似不保证事实相同。配对共享题目和金标，但答案/裁判均是独立运行。",
        "arms": arms,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: arm["counts"] for name, arm in arms.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
