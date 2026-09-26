#!/usr/bin/env python3
"""筛中文历史图时间题中适合做“补日期”小 POC 的真实候选；零模型调用。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import audit_graph_context_zh as previous

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/analysis/m2_date_poc_eligibility_20260926.json"
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def main() -> None:
    rows = previous.load(previous.RAW_GRAPH)
    grades = previous.load(previous.GRADE_GRAPH)
    vector_by_id = {memory["id"]: memory for row in rows.values()
                    for memory in row["retrieved_memories"]}
    counts = Counter()
    no_year_ids = set()
    cases = []
    for case_id in sorted(set(rows) & set(grades)):
        if grades[case_id]["category"] != "temporal":
            continue
        counts["temporal_cases"] += 1
        row = rows[case_id]
        graph_by_id = {memory["id"]: memory for memory in row["graph_memories"]}
        selected = row["metadata"]["graph_context_ids"]
        if not selected:
            continue
        counts["graph_selected_cases"] += 1
        no_year = []
        for memory_id in selected:
            memory = graph_by_id[memory_id]
            source = vector_by_id.get(memory_id)
            date = (source.get("metadata") or {}).get("session_date") if source else None
            if source and source["content"] != memory["content"]:
                raise ValueError(f"content differs: {memory_id}")
            explicit_year = bool(YEAR.search(memory["content"]))
            counts["selected_graph_instances"] += 1
            counts["date_recoverable_instances"] += bool(date)
            counts["content_has_explicit_four_digit_year"] += explicit_year
            if explicit_year:
                continue
            counts["content_without_four_digit_year"] += 1
            no_year_ids.add(memory_id)
            no_year.append({
                "id": memory_id, "content": memory["content"],
                "recovered_session_date": date,
            })
        if no_year:
            counts["cases_with_no_year_graph_memory"] += 1
            counts["nonexact_cases_with_no_year_graph_memory"] += grades[case_id]["final"] != "exact"
            cases.append({
                "case_id": case_id,
                "question": row["question"],
                "gold": row["expected_answer"],
                "answer": row["predicted_answer"],
                "grade": grades[case_id]["final"],
                "no_year_selected_graph": no_year,
            })
    counts["unique_no_year_graph_ids"] = len(no_year_ids)
    report = {
        "scope": "2026-09-25 中文融合图臂的 J 类时间题最终入选图记忆",
        "method": "按最终 answer_context_ids 对应的 graph_context_ids，检查内容是否直接含四位年份、同 ID 原记忆日期是否可恢复；保留无年份候选供人工判断事实相关性。",
        "limits": "无四位年份不代表需要会话日期；有四位年份也未必与当前问题相关或正确。人工语义核查不能由这些词面计数替代。",
        "counts": dict(counts),
        "cases_with_no_year_graph_memory": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
