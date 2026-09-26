#!/usr/bin/env python3
"""从中文源 turn 出发审计非 exact 时间题中的相对日词；零模型调用。"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
RAW = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
GRADES = ROOT / "results/precision_regrade_zhfull.jsonl"
OUTPUT = ROOT / "results/analysis/m2_relative_day_funnel_20260926.json"
RELATIVE_DAYS = {"昨天": -1, "前天": -2, "明天": 1, "后天": 2}
RELATIVE_PATTERN = re.compile("|".join(RELATIVE_DAYS))
SESSION_PATTERN = re.compile(r"session_\d+")


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            case_id = row["case_id"]
            if case_id in rows:
                raise ValueError(f"duplicate case ID: {path}: {case_id}")
            rows[case_id] = row
    return rows


def session_date(value: str) -> str:
    return datetime.strptime(value.split(" on ", 1)[1], "%d %B, %Y").date().isoformat()


def main() -> None:
    dataset = json.loads(DATASET.read_text())
    raw, grades = load_jsonl(RAW), load_jsonl(GRADES)
    counts = Counter()
    cases = []
    for entry in dataset:
        conv_id = entry["sample_id"]
        conversation = entry["conversation"]
        by_id = {
            turn["dia_id"]: (session_id, index, turn)
            for session_id, turns in conversation.items()
            if SESSION_PATTERN.fullmatch(session_id) and isinstance(turns, list)
            for index, turn in enumerate(turns)
        }
        for qa_index, qa in enumerate(entry["qa"]):
            case_id = f"locomo_{conv_id}_qa{qa_index}"
            if case_id not in raw or case_id not in grades:
                continue
            grade = grades[case_id]
            if grade["category"] != "temporal":
                continue
            counts["temporal_J_cases"] += 1
            if grade["final"] == "exact":
                continue
            counts["nonexact_temporal_J_cases"] += 1
            evidence = qa.get("evidence") or []
            hits = []
            for dia_id in evidence:
                if dia_id not in by_id:
                    continue
                session_id, index, turn = by_id[dia_id]
                text = turn.get("text", "")
                tokens = sorted(set(RELATIVE_PATTERN.findall(text)))
                if not tokens:
                    continue
                anchor = session_date(conversation[f"{session_id}_date_time"])
                turns = conversation[session_id]
                hits.append({
                    "dia_id": dia_id,
                    "session_id": session_id,
                    "session_date": anchor,
                    "tokens": tokens,
                    "arithmetic_dates": {
                        token: (datetime.fromisoformat(anchor).date() + timedelta(days=RELATIVE_DAYS[token])).isoformat()
                        for token in tokens
                    },
                    "turn": turn,
                    "neighbor_turns": turns[max(0, index - 2):min(len(turns), index + 3)],
                })
            if not hits:
                continue
            counts["cases_with_relative_day_in_gold_evidence"] += 1
            counts[f"grade_{grade['final']}"] += 1
            row = raw[case_id]
            selected_ids = set(row["metadata"]["answer_context_ids"])
            memories = row["retrieved_memories"]
            if selected_ids != {memory["id"] for memory in memories}:
                raise AssertionError(f"unexpected answer context: {case_id}")
            source_sessions = {hit["session_id"] for hit in hits}
            cases.append({
                "case_id": case_id,
                "question": qa["question"],
                "gold": qa["answer"],
                "all_gold_evidence_ids": evidence,
                "all_gold_evidence_turns": [
                    {"dia_id": dia_id, "session_id": by_id[dia_id][0],
                     "session_date": session_date(conversation[f"{by_id[dia_id][0]}_date_time"]),
                     "turn": by_id[dia_id][2]}
                    for dia_id in evidence if dia_id in by_id
                ],
                "grade": grade["final"],
                "strict_judge": grade["judge"],
                "answer": row["predicted_answer"],
                "relative_day_source_hits": hits,
                "same_source_session_memories_in_top15": [
                    {"rank": rank, **memory}
                    for rank, memory in enumerate(memories, 1)
                    if (memory.get("metadata") or {}).get("session_id") in source_sessions
                ],
            })
    report = {
        "scope": "2026-09-25 中文无图归档、严格非 exact 的 temporal J 题；金标 evidence turn 直接含昨天/前天/明天/后天",
        "method": "只按词面与会话日期做算术筛选；逐题保留源 turn、邻近 turn、同来源会话的 top-15 记忆、答案和严格裁判。词面命中不保证与所问事件相关。",
        "source_dataset": str(DATASET),
        "raw_archive": str(RAW.relative_to(ROOT)),
        "grades": str(GRADES.relative_to(ROOT)),
        "counts": dict(counts),
        "cases": cases,
        "limits": [
            "会话日期是提及日；算术日期仅针对命中的词，不等于所问事件日期。",
            "gold evidence 可能错标说话者、事件或年份；必须逐题核对。",
            "只筛四个相对日词，不能估计全部 temporal 错误的机制比例。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
