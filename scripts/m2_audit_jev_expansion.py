#!/usr/bin/env python3
"""复算中文 Jev 扩拉的候选进出账；只读归档，不调用模型。

同夜无 Jev 图臂只近似首判候选池，不能当作真实首判快照。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import audit_jev_expansion_zh as previous

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/analysis/m2_jev_expansion_20260926.json"


def audit_arm(raw_path: Path, grade_path: Path, control: dict, control_grade: dict) -> dict:
    raw, grades = previous.load(raw_path), previous.load(grade_path)
    groups: dict[str, Counter] = defaultdict(Counter)
    by_category: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    new_id_wins = []
    for case_id in sorted(set(control) & set(raw) & set(control_grade) & set(grades)):
        category = control_grade[case_id]["category"]
        if category not in previous.CATEGORIES:
            continue
        left, right = control[case_id], raw[case_id]
        stop = right["metadata"].get("jev_stop") or {}
        if not isinstance(stop.get("first"), (float, int)):
            continue
        expanded = bool(stop.get("expanded"))
        left_vectors = {m["id"]: m for m in left["retrieved_memories"]}
        left_graph = {m["id"]: m for m in left["graph_memories"]}
        right_vectors = {m["id"]: m for m in right["retrieved_memories"]}
        right_graph = {m["id"]: m for m in right["graph_memories"]}
        left_pool = set(left_vectors) | set(left_graph)
        final_ids = right["metadata"]["answer_context_ids"]
        new_ids = [memory_id for memory_id in final_ids if memory_id not in left_pool]
        group = ("expanded_new_id" if new_ids else "expanded_old_pool") if expanded else "not_expanded"
        left_exact = control_grade[case_id]["final"] == "exact"
        right_exact = grades[case_id]["final"] == "exact"
        outcome = "jev_only" if right_exact and not left_exact else "control_only" if left_exact and not right_exact else "same"
        same_context = previous.context(left) == previous.context(right)
        for counter in (groups[group], by_category[category][group]):
            counter["cases"] += 1
            counter[outcome] += 1
            counter["new_selected_ids"] += len(new_ids)
            counter["new_selected_graph_ids"] += sum(memory_id in right_graph for memory_id in new_ids)
            counter["new_selected_vector_ids"] += sum(memory_id in right_vectors for memory_id in new_ids)
            counter["same_vector_candidates"] += list(left_vectors) == list(right_vectors)
            counter["same_full_context"] += same_context
            counter["same_context_jev_only"] += same_context and outcome == "jev_only"
            counter["same_context_control_only"] += same_context and outcome == "control_only"
            counter["new_graph_candidates"] += len(set(right_graph) - set(left_graph))
            counter["old_graph_candidates_lost"] += len(set(left_graph) - set(right_graph))
            counter["control_context_ids_displaced"] += len(
                set(left["metadata"]["answer_context_ids"]) - set(final_ids)
            )
            counter["second_below_0_4"] += expanded and stop.get("second", 1) < 0.4
        if expanded and new_ids and outcome == "jev_only":
            records = right_vectors | right_graph
            new_id_wins.append({
                "case_id": case_id,
                "category": category,
                "question": right["question"],
                "gold": right["expected_answer"],
                "control_answer": left["predicted_answer"],
                "jev_answer": right["predicted_answer"],
                "control_judge": control_grade[case_id]["judge"],
                "jev_judge": grades[case_id]["judge"],
                "first": stop["first"],
                "second": stop.get("second"),
                "new_selected": [
                    {
                        "id": memory_id,
                        "source": "graph" if memory_id in right_graph else "vector",
                        "content": records[memory_id]["content"],
                    }
                    for memory_id in new_ids
                ],
            })
    return {
        "groups": {key: dict(value) for key, value in sorted(groups.items())},
        "by_category": {
            category: {key: dict(value) for key, value in sorted(group.items())}
            for category, group in sorted(by_category.items())
        },
        "new_id_jev_only_cases": new_id_wins,
    }


def main() -> None:
    control, control_grade = previous.load(previous.CONTROL), previous.load(previous.CONTROL_GRADE)
    result = {
        "scope": "2026-09-25 中文同夜融合图/Jev 归档，严格重判 J 类共同题",
        "method": "以无 Jev 臂同题向量+图候选 ID 并集近似未记录的 Jev 首判池；按最终 answer_context_ids 核对是否选入池外 ID。",
        "limits": "两臂为独立运行。未扩拉题也会有少量池外 ID；池外 ID 不是新事实，扩拉后无池外入选也不等于没有取回候选。分组是干预后观测，翻分不可作因果效应。",
        "arms": {
            name: audit_arm(*paths, control, control_grade)
            for name, paths in previous.ARMS.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: arm["groups"] for name, arm in result["arms"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
