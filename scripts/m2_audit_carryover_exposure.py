#!/usr/bin/env python3
"""只读配对旧中文四臂，核对跨会话旧事实写入后的实际入槽来源。"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/analysis/m2_cross_session_carryover_packet_20260926.json"
LABELS = ROOT / "results/analysis/m2_cross_session_carryover_labels_20260926.json"
OUTPUT = ROOT / "results/analysis/m2_carryover_exposure_20260926.json"
ARMS = {
    "无图": ("results_zh/zhfull_locomo_20260925_115806.jsonl", "results/precision_regrade_zhfull.jsonl"),
    "融合图": ("results_zh/zhfull_fused_locomo_20260925_195248.jsonl", "results/precision_regrade_fused.jsonl"),
    "Jev_decider": ("results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl", "results/precision_regrade_jevdec.jsonl"),
    "Jev_laya": ("results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl", "results/precision_regrade_jevlaya.jsonl"),
}
J_CATEGORIES = {"single-hop", "multi-hop", "temporal", "open-domain"}


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            if row["case_id"] in rows:
                raise ValueError(f"duplicate case ID in {path}: {row['case_id']}")
            rows[row["case_id"]] = row
    return rows


def parsed_date(memory: dict) -> datetime:
    value = memory["metadata"]["session_date"]
    return datetime.strptime(value.split(" on ", 1)[1], "%d %B, %Y")


def main() -> None:
    packet = json.loads(PACKET.read_text())
    labels = {row["pair_id"]: row for row in json.loads(LABELS.read_text())["逐对"]}
    stale_pairs = []
    for pair in packet["pairs"]:
        if labels[pair["pair_id"]]["分类"] != "旧事实写入较晚场次_未见新来源":
            continue
        left, right = pair["left"], pair["right"]
        if parsed_date(left) == parsed_date(right):
            raise AssertionError(f"same-day pair has no later session: {pair['pair_id']}")
        earlier, later = (left, right) if parsed_date(left) < parsed_date(right) else (right, left)
        stale_pairs.append({
            "pair_id": pair["pair_id"], "conversation": pair["conversation"],
            "earlier_id": earlier["id"], "earlier_session": earlier["metadata"]["session_id"],
            "later_id": later["id"], "later_session": later["metadata"]["session_id"],
        })
    later_ids = {pair["later_id"] for pair in stale_pairs}
    raw = {name: load_jsonl(ROOT / paths[0]) for name, paths in ARMS.items()}
    grades = {name: load_jsonl(ROOT / paths[1]) for name, paths in ARMS.items()}
    common = set.intersection(*(set(raw[name]) & set(grades[name]) for name in ARMS))
    common = sorted(case_id for case_id in common
                    if grades["无图"][case_id]["category"] in J_CATEGORIES)
    counts = {name: Counter() for name in ARMS}
    exposed = {name: set() for name in ARMS}
    cases = []
    for case_id in common:
        case_arms = {}
        for name in ARMS:
            row = raw[name][case_id]
            context = set(row["metadata"]["answer_context_ids"])
            vector = {memory["id"] for memory in row["retrieved_memories"]}
            graph = {memory["id"] for memory in row["graph_memories"]}
            selected_later = sorted(context & later_ids)
            pair_ids = [pair["pair_id"] for pair in stale_pairs
                        if pair["earlier_id"] in context and pair["later_id"] in context]
            if selected_later:
                exposed[name].add(case_id)
                counts[name]["exposed_cases"] += 1
                counts[name]["later_memory_slots"] += len(selected_later)
                counts[name]["later_via_vector_slots"] += len(set(selected_later) & vector)
                counts[name]["later_via_graph_slots"] += len(set(selected_later) & graph)
                counts[name][f"exposed_grade_{grades[name][case_id]['final']}"] += 1
            counts[name]["pair_co_selected_cases"] += bool(pair_ids)
            if selected_later or pair_ids:
                case_arms[name] = {
                    "later_ids": selected_later,
                    "later_via_graph_ids": sorted(set(selected_later) & graph),
                    "co_selected_pair_ids": pair_ids,
                    "strict_grade": grades[name][case_id]["final"],
                }
            if len(row["retrieved_memories"]) > 15:
                raise AssertionError(f"unexpected saved vector count: {case_id}, {name}")
        if case_arms:
            cases.append({"case_id": case_id, "category": grades["无图"][case_id]["category"],
                          "arms": case_arms})
    transitions = {}
    for name in ARMS:
        if name == "无图":
            continue
        transitions[name] = {
            "both_exposed": len(exposed["无图"] & exposed[name]),
            "base_only": len(exposed["无图"] - exposed[name]),
            "arm_only": len(exposed[name] - exposed["无图"]),
        }
    report = {
        "scope": "四个 2026-09-25 中文归档共同且四臂均有严格重判的 J 类题；仅用人工回源标为旧事实跨场写入的 12 对记忆",
        "common_J_cases": len(common),
        "stale_pairs": stale_pairs,
        "arm_counts": {name: dict(counts[name]) for name in ARMS},
        "vs_no_graph_transitions": transitions,
        "cases_with_any_exposure": cases,
        "limits": [
            "这些是固定高相似候选中的 12 条较晚记忆，不是全部错误来源记忆。",
            "图/Jev 与无图为独立运行；不同臂的入槽变化不能直接解释答题因果效应。",
            "归档向量检索只保存最终 top-15，未保留预重排 top-45 与得分；不能零调用推断删去重复记忆后的第 16 名或答案。",
            "Jev 扩拉后的 graph_context_ids 可过期；图来源按最终 answer_context_ids 与 graph_memories ID 交集计算。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"common_J_cases": report["common_J_cases"],
                      "arm_counts": report["arm_counts"],
                      "vs_no_graph_transitions": transitions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
