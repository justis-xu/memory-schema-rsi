#!/usr/bin/env python3
"""为中文错题的事实级漏斗准备固定分层样本与源 turn/记忆证据包；零模型调用。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
SESSION_AUDIT = ROOT / "results/analysis/evidence_session_audit_20260926.json"
RAW = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
GRADES = ROOT / "results/precision_regrade_zhfull.jsonl"
OUTPUT = ROOT / "results/analysis/m2_fact_funnel_packet_20260926.json"
EXCLUDED_CASES = {
    "locomo_conv-26_qa36", "locomo_conv-26_qa41", "locomo_conv-30_qa12",
    "locomo_conv-30_qa15", "locomo_conv-30_qa31", "locomo_conv-30_qa33",
    "locomo_conv-43_qa16", "locomo_conv-43_qa24", "locomo_conv-43_qa56",
    "locomo_conv-43_qa6", "locomo_conv-43_qa60", "locomo_conv-43_qa63",
    "locomo_conv-44_qa5", "locomo_conv-44_qa55", "locomo_conv-44_qa57",
    "locomo_conv-44_qa58", "locomo_conv-44_qa6",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            if row["case_id"] in rows:
                raise ValueError(f"duplicate case ID: {row['case_id']}")
            rows[row["case_id"]] = row
    return rows


def main() -> None:
    dataset = json.loads(DATASET.read_text())
    qa_by_id = {
        f"locomo_{entry['sample_id']}_qa{i}": qa
        for entry in dataset for i, qa in enumerate(entry["qa"])
    }
    turns_by_conv = {}
    for entry in dataset:
        by_dia = {}
        for key, turns in entry["conversation"].items():
            if re.fullmatch(r"session_\d+", key) and isinstance(turns, list):
                for index, turn in enumerate(turns):
                    by_dia[turn["dia_id"]] = {
                        "session_id": key,
                        "session_date": entry["conversation"].get(f"{key}_date_time"),
                        "index": index,
                        "turn": turn,
                    }
        turns_by_conv[entry["sample_id"]] = by_dia
    raw = load_jsonl(RAW)
    grades = load_jsonl(GRADES)
    coarse = {row["case_id"]: row for row in json.loads(SESSION_AUDIT.read_text())["cases"]}
    memory_pool = defaultdict(dict)
    for case_id, row in raw.items():
        conv_id = case_id.split("_")[1]
        for memory in row["retrieved_memories"]:
            memory_pool[conv_id][memory["id"]] = memory

    strata = defaultdict(list)
    for case_id, row in coarse.items():
        if case_id in EXCLUDED_CASES or row["strict_final"] == "exact":
            continue
        if row["stage"] != "all_gold_sessions_in_top15":
            continue
        conv_id = case_id.split("_")[1]
        evidence = qa_by_id[case_id].get("evidence") or []
        if not evidence or any(
            not re.fullmatch(r"D\d+:\d+", str(dia)) or dia not in turns_by_conv[conv_id]
            for dia in evidence
        ):
            continue
        strata[(conv_id, grades[case_id]["category"])].append(case_id)

    selected = [
        min(case_ids, key=lambda cid: hashlib.sha256(cid.encode()).hexdigest())
        for _, case_ids in sorted(strata.items())
    ]
    cases = []
    for case_id in selected:
        conv_id = case_id.split("_")[1]
        qa = qa_by_id[case_id]
        source = [turns_by_conv[conv_id][dia] for dia in qa["evidence"]]
        gold_sessions = {item["session_id"] for item in source}
        context = raw[case_id]["retrieved_memories"]
        context_ids = set(raw[case_id]["metadata"]["answer_context_ids"])
        if context_ids != {memory["id"] for memory in context}:
            raise AssertionError(f"unexpected context: {case_id}")
        pool_same_session = [
            memory for memory in memory_pool[conv_id].values()
            if (memory.get("metadata") or {}).get("session_id") in gold_sessions
            and memory["id"] not in context_ids
        ]
        cases.append({
            "case_id": case_id,
            "stratum": {"conversation": conv_id, "category": grades[case_id]["category"]},
            "question": qa["question"], "gold": qa["answer"],
            "answer": raw[case_id]["predicted_answer"],
            "strict_final": grades[case_id]["final"], "strict_judge": grades[case_id]["judge"],
            "gold_evidence_turns": source,
            "top15_memories": [
                {"rank": index + 1, "same_gold_session": (memory.get("metadata") or {}).get("session_id") in gold_sessions,
                 "id": memory["id"], "content": memory["content"], "metadata": memory.get("metadata")}
                for index, memory in enumerate(context)
            ],
            "other_archived_retrieved_memories_from_gold_sessions": pool_same_session,
        })
    report = {
        "scope": "中文历史无图首轮，来源 session 已在 top-15 但严格非 exact 的题；按对话×题型固定哈希选一题，原始 evidence ID 必须存在",
        "selection": "每个非空对话×题型格选 SHA-256(case_id) 最小的一题；排除已在时间金标专题详查的题与 conv-30_qa31。样本用于机制诊断，不估计总体占比。",
        "raw_archive": str(RAW.relative_to(ROOT)),
        "grades": str(GRADES.relative_to(ROOT)),
        "source_dataset": str(DATASET),
        "candidate_strata": {f"{conv}/{cat}": len(ids) for (conv, cat), ids in sorted(strata.items())},
        "selected_case_ids": selected,
        "cases": cases,
        "limits": [
            "other_archived_retrieved_memories 是所有题 top-15 的并集，不等于完整历史 Mem0 库。",
            "gold evidence turn 可能不完整或语义错标；人工逐题核查，不将 session 出现当成事实出现。",
            "这批题按严格非 exact 条件选择，只能诊断错误类型，不能估算全量错题的机制比例。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_strata": report["candidate_strata"], "selected_case_ids": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
