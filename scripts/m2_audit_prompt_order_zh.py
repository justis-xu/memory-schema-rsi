#!/usr/bin/env python3
"""复算中文 LoCoMo 答题提示中的日期展示与实际排序；零模型调用。"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema_rsi.benchmarks.base import parse_session_date  # noqa: E402
from schema_rsi.evaluation.prompts_official import format_memories_official  # noqa: E402
from schema_rsi.memory.base import MemoryRecord  # noqa: E402

import audit_graph_context_zh as previous  # noqa: E402

OUTPUT = ROOT / "results/analysis/m2_prompt_order_20260926.json"


def selected(row: dict) -> list[dict]:
    vectors = {m["id"]: m for m in row["retrieved_memories"]}
    graphs = {m["id"]: m for m in row.get("graph_memories", [])}
    out = []
    for memory_id in row["metadata"]["answer_context_ids"]:
        if memory_id in graphs:
            record = graphs[memory_id]
            # pipeline.py constructs a fresh MemoryRecord with via/user_id only.
            date = None
            graph = True
        else:
            record = vectors[memory_id]
            date = (record.get("metadata") or {}).get("session_date")
            graph = False
        out.append({"id": memory_id, "content": record["content"], "date": date, "graph": graph})
    return out


def ordered(records: list[dict], *, restored: dict[str, str] | None = None,
            parse_dates: bool = False) -> list[dict]:
    def key(record: dict) -> str:
        date = restored.get(record["id"], "") if restored and record["graph"] else record["date"]
        if parse_dates:
            return parse_session_date(date) or "9999"
        return date or ""

    return sorted(records, key=key)


def ids(records: list[dict]) -> list[str]:
    return [record["id"] for record in records]


def date_head(records: list[dict], restored: dict[str, str] | None = None, limit: int = 5) -> list[dict]:
    return [
        {
            "id": record["id"],
            "graph": record["graph"],
            "date": restored.get(record["id"], "") if restored and record["graph"] else record["date"],
        }
        for record in records[:limit]
    ]


def audit_arm(rows: dict, grades: dict, vector_dates: dict[str, str], *, graph_arm: bool) -> dict:
    counts = Counter()
    by_category = defaultdict(Counter)
    examples = []
    for case_id in sorted(set(rows) & set(grades)):
        category = grades[case_id]["category"]
        if category not in previous.CATEGORIES:
            continue
        row = rows[case_id]
        records = selected(row)
        if any(not r["graph"] and not r["date"] for r in records):
            raise ValueError(f"selected vector memory has no session_date: {case_id}")
        restored = {r["id"]: vector_dates[r["id"]] for r in records if r["graph"] and r["id"] in vector_dates}
        current = ordered(records)
        date_only = ordered(records, restored=restored)
        chronological = ordered(records, restored=restored, parse_dates=True)
        known_dates = [parse_session_date(r["date"]) for r in current if r["date"]]
        if any(date is None for date in known_dates):
            raise ValueError(f"unparseable selected date: {case_id}")
        inversions = sum(known_dates[i] > known_dates[i + 1] for i in range(len(known_dates) - 1))
        selected_graph = [r for r in records if r["graph"]]
        if counts["renderer_order_checked"] < 100 and all("\n" not in r["content"] for r in records):
            vectors = {m["id"]: m for m in row["retrieved_memories"]}
            as_memory_records = [
                MemoryRecord(
                    id=r["id"], content=r["content"],
                    metadata={"via": [], "user_id": row["metadata"]["user_id"]}
                    if r["graph"] else vectors[r["id"]]["metadata"],
                )
                for r in records
            ]
            rendered_lines = format_memories_official(as_memory_records).splitlines()[2:]
            if len(rendered_lines) != len(current) or any(
                    not line.endswith(r["content"]) for line, r in zip(rendered_lines, current)):
                raise ValueError(f"renderer order differs: {case_id}")
            counts["renderer_order_checked"] += 1
        for counter in (counts, by_category[category]):
            counter["cases"] += 1
            counter["raw_date_nonchronological_cases"] += inversions > 0
            counter["raw_date_adjacent_inversions"] += inversions
            counter["parsed_chronological_order_changes"] += ids(current) != ids(chronological)
            counter["graph_selected_cases"] += bool(selected_graph)
            counter["graph_selected_instances"] += len(selected_graph)
            counter["graph_dates_recovered"] += len(restored)
            if selected_graph:
                graph_count = len(selected_graph)
                counter["all_graph_lines_before_dated_vectors"] += all(r["graph"] for r in current[:graph_count])
                counter["date_restore_changes_order"] += ids(current) != ids(date_only)
                counter["chronological_fix_changes_order_after_restore"] += ids(date_only) != ids(chronological)
                counter["all_graph_dates_recovered_cases"] += len(restored) == graph_count
        if (graph_arm and category == "temporal" and selected_graph and len(restored) == len(selected_graph)
                and len(examples) < 8):
            examples.append({
                "case_id": case_id,
                "question": row["question"],
                "gold": row["expected_answer"],
                "archive_answer": row["predicted_answer"],
                "graph_selected": len(selected_graph),
                "current_prompt_head": date_head(current),
                "date_restored_head": date_head(date_only, restored),
                "chronological_head": date_head(chronological, restored),
            })
    return {
        "counts": dict(counts),
        "by_category": {key: dict(value) for key, value in sorted(by_category.items())},
        "examples": examples,
    }


def main() -> None:
    baseline, graph = previous.load(previous.RAW_BASE), previous.load(previous.RAW_GRAPH)
    base_grades, graph_grades = previous.load(previous.GRADE_BASE), previous.load(previous.GRADE_GRAPH)
    vector_dates = {}
    vector_content = {}
    for row in graph.values():
        for record in row["retrieved_memories"]:
            memory_id = record["id"]
            date = (record.get("metadata") or {}).get("session_date")
            if memory_id in vector_content and vector_content[memory_id] != record["content"]:
                raise ValueError(f"content changed for {memory_id}")
            vector_content[memory_id] = record["content"]
            if date:
                vector_dates[memory_id] = date
    for row in graph.values():
        for record in row.get("graph_memories", []):
            memory_id = record["id"]
            if memory_id in vector_content and vector_content[memory_id] != record["content"]:
                raise ValueError(f"graph/vector content differs for {memory_id}")
    result = {
        "scope": "2026-09-25 中文无图与融合图归档 J 类答题上下文",
        "method": "镜像 prompts_official.format_memories_official 的原始 session_date 字符串排序；再对比仅继承图日期，以及解析为 ISO 后真正按时间排序。",
        "limits": "只验证提示输入的顺序和日期，不测答案效应。日期继承使用同次归档其他向量召回的相同 ID/内容；两种修正仅作离线对照。",
        "arms": {
            "baseline": audit_arm(baseline, base_grades, vector_dates, graph_arm=False),
            "graph": audit_arm(graph, graph_grades, vector_dates, graph_arm=True),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: arm["counts"] for name, arm in result["arms"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
