#!/usr/bin/env python3
"""核中文图臂时间题翻分与显式年份问句的金标源会话；零模型调用。"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema_rsi.benchmarks.base import parse_session_date  # noqa: E402

import audit_graph_context_zh as previous  # noqa: E402

ZH = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
EN = ROOT / "data/locomo/locomo10.json"
OUTPUT = ROOT / "results/analysis/m2_temporal_flip_sources_20260926.json"
YEAR = re.compile(r"(?<!\d)202[234](?!\d)")
TURN_ID = re.compile(r"D\d+:\d+")


def source_turns(entry: dict) -> dict[str, dict]:
    result = {}
    conversation = entry["conversation"]
    for key, turns in conversation.items():
        if not re.fullmatch(r"session_\d+", key):
            continue
        date_raw = conversation.get(f"{key}_date_time")
        date_iso = parse_session_date(date_raw)
        for turn in turns:
            result[turn["dia_id"]] = {
                "session": key, "session_date_raw": date_raw,
                "session_date_iso": date_iso, "text": turn["text"],
            }
    return result


def year_screen() -> dict:
    zh = {entry["sample_id"]: entry for entry in json.loads(ZH.read_text())}
    en = {entry["sample_id"]: entry for entry in json.loads(EN.read_text())}
    if zh.keys() != en.keys():
        raise ValueError("Chinese/English conversations differ")
    counts = Counter()
    candidates = []
    for conv, entry in sorted(zh.items()):
        source = source_turns(entry)
        original = source_turns(en[conv])
        if len(entry["qa"]) != len(en[conv]["qa"]):
            raise ValueError(f"Chinese/English QA count differs: {conv}")
        for index, question in enumerate(entry["qa"]):
            if question["category"] not in (1, 2, 3, 4):
                continue
            counts["J_cases"] += 1
            years = set(YEAR.findall(question["question"]))
            if not years:
                continue
            counts["J_explicit_2022_2024_year"] += 1
            evidence_ids = [match.group() for item in question.get("evidence", [])
                            for match in TURN_ID.finditer(str(item))]
            if not evidence_ids or any(evidence_id not in source for evidence_id in evidence_ids):
                continue
            if any(not source[evidence_id]["session_date_iso"] for evidence_id in evidence_ids):
                continue
            evidence_years = {source[evidence_id]["session_date_iso"][:4]
                              for evidence_id in evidence_ids}
            counts["J_explicit_year_with_parseable_evidence"] += 1
            if not years.isdisjoint(evidence_years):
                continue
            counts["question_year_disjoint_from_evidence_session_year"] += 1
            en_question = en[conv]["qa"][index]
            candidates.append({
                "case_id": f"locomo_{conv}_qa{index}",
                "category": question["category"],
                "question_zh": question["question"],
                "gold_zh": question.get("answer"),
                "question_en": en_question["question"],
                "gold_en": en_question.get("answer"),
                "question_years": sorted(years),
                "evidence_session_years": sorted(evidence_years),
                "evidence_ids": evidence_ids,
                "source_zh": {memory_id: source[memory_id] for memory_id in evidence_ids},
                "source_en": {memory_id: original[memory_id] for memory_id in evidence_ids},
            })
    return {"counts": dict(counts), "candidates": candidates}


def temporal_graph_flips() -> dict:
    baseline, graph = previous.load(previous.RAW_BASE), previous.load(previous.RAW_GRAPH)
    base_grade, graph_grade = previous.load(previous.GRADE_BASE), previous.load(previous.GRADE_GRAPH)
    vector_by_id = {m["id"]: m for row in graph.values() for m in row["retrieved_memories"]}
    counts = Counter()
    cases = []
    for case_id in sorted(set(baseline) & set(graph) & set(base_grade) & set(graph_grade)):
        if base_grade[case_id]["category"] != "temporal":
            continue
        counts["paired_temporal"] += 1
        left, right = baseline[case_id], graph[case_id]
        graph_ids = right["metadata"]["graph_context_ids"]
        if not graph_ids:
            continue
        counts["graph_selected_temporal"] += 1
        left_exact = base_grade[case_id]["final"] == "exact"
        right_exact = graph_grade[case_id]["final"] == "exact"
        if left_exact == right_exact:
            continue
        outcome = "graph_only" if right_exact else "baseline_only"
        counts[outcome] += 1
        left_vectors = {m["id"]: m for m in left["retrieved_memories"]}
        right_graph = {m["id"]: m for m in right["graph_memories"]}
        cases.append({
            "case_id": case_id,
            "outcome": outcome,
            "question": right["question"],
            "gold": right["expected_answer"],
            "baseline_answer": left["predicted_answer"],
            "graph_answer": right["predicted_answer"],
            "baseline_judge": base_grade[case_id]["judge"],
            "graph_judge": graph_grade[case_id]["judge"],
            "same_vector_candidate_ids": [m["id"] for m in left["retrieved_memories"]]
                                         == [m["id"] for m in right["retrieved_memories"]],
            "selected_graph": [
                {
                    "id": memory_id,
                    "content": right_graph[memory_id]["content"],
                    "recovered_session_date": (vector_by_id.get(memory_id, {}).get("metadata") or {}).get("session_date"),
                }
                for memory_id in graph_ids
            ],
            "baseline_context_displaced": [
                {"id": memory_id, "content": left_vectors[memory_id]["content"]}
                for memory_id in left["metadata"]["answer_context_ids"]
                if memory_id not in right["metadata"]["answer_context_ids"]
            ],
        })
    return {"counts": dict(counts), "cases": cases}


def main() -> None:
    report = {
        "scope": "中文融合图时间题翻分与中文 LoCoMo 2022-2024 显式年份问句的源会话核查",
        "limits": "年份不一致仅是候选筛选；回忆往事、未来计划和对抗题需人工区分。翻分案例是干预后选出的观察，不可按赢/输推因果。",
        "year_screen": year_screen(),
        "temporal_graph_flips": temporal_graph_flips(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value["counts"] for key, value in report.items() if isinstance(value, dict) and "counts" in value}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
