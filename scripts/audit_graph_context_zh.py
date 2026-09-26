#!/usr/bin/env python3
"""Descriptive paired audit of Chinese baseline and fused-graph answer contexts.

Post-treatment graph-selection strata are descriptive, not causal estimates.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_BASE = ROOT / "results_zh/zhfull_locomo_20260925_164645.jsonl"
RAW_GRAPH = ROOT / "results_zh/zhfull_fused_locomo_20260925_172126.jsonl"
GRADE_BASE = ROOT / "results/precision_regrade_base.jsonl"
GRADE_GRAPH = ROOT / "results/precision_regrade_graph.jsonl"
OUTPUT = ROOT / "results/analysis/graph_context_audit_20260926.json"
CATEGORIES = {"single-hop", "multi-hop", "temporal", "open-domain"}


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if row["case_id"] in rows:
            raise ValueError(f"duplicate case_id: {path}, {row['case_id']}")
        rows[row["case_id"]] = row
    return rows


def fingerprint(row: dict) -> list[tuple[str, str | None, str | None]]:
    records = {m["id"]: m for m in row.get("retrieved_memories", []) + row.get("graph_memories", [])}
    out = []
    for memory_id in row["metadata"]["answer_context_ids"]:
        record = records.get(memory_id)
        if record is None:
            raise ValueError(f"missing context record: {row['case_id']}, {memory_id}")
        out.append((memory_id, record.get("content"), (record.get("metadata") or {}).get("session_date")))
    return out


def main() -> None:
    baseline, graph = load(RAW_BASE), load(RAW_GRAPH)
    base_grade, graph_grade = load(GRADE_BASE), load(GRADE_GRAPH)
    vector_by_id = {}
    for row in graph.values():
        for memory in row.get("retrieved_memories", []):
            prior = vector_by_id.get(memory["id"])
            if prior is not None and prior["content"] != memory["content"]:
                raise ValueError(f"inconsistent vector memory content: {memory['id']}")
            vector_by_id[memory["id"]] = memory
    common = sorted(set(baseline) & set(graph) & set(base_grade) & set(graph_grade))
    counts = Counter()
    selection_distribution = Counter()
    by_stratum = defaultdict(Counter)
    by_category = defaultdict(lambda: defaultdict(Counter))
    cases = []
    for case_id in common:
        category = base_grade[case_id]["category"]
        if category not in CATEGORIES:
            continue
        a, b = baseline[case_id], graph[case_id]
        a_ids = a["metadata"]["answer_context_ids"]
        b_ids = b["metadata"]["answer_context_ids"]
        g_ids = b["metadata"]["graph_context_ids"]
        assert len(g_ids) == len(set(g_ids))
        assert set(g_ids) <= set(b_ids)
        graph_records = {m["id"]: m for m in b.get("graph_memories", [])}
        assert set(g_ids) <= graph_records.keys()
        a_retrieved = {m["id"] for m in a["retrieved_memories"]}
        b_retrieved = {m["id"] for m in b["retrieved_memories"]}
        assert not set(g_ids) & b_retrieved
        novel_ids = [memory_id for memory_id in g_ids if memory_id not in a_retrieved]
        configured_slots = b["metadata"]["graph_slots"]
        selection_distribution[len(g_ids)] += 1
        stratum = "no_graph_selected" if not g_ids else (
            "selected_new_to_baseline_candidates" if novel_ids else "selected_seen_in_baseline_candidates"
        )
        left = base_grade[case_id]["final"] == "exact"
        right = graph_grade[case_id]["final"] == "exact"
        outcome = "graph_only" if right and not left else "baseline_only" if left and not right else "both_exact" if left else "neither_exact"
        same_ids = a_ids == b_ids
        same_fingerprint = fingerprint(a) == fingerprint(b) if same_ids else False
        lost = [memory_id for memory_id in a_ids if memory_id not in b_ids]
        gained = [memory_id for memory_id in b_ids if memory_id not in a_ids]
        counts["cases"] += 1
        counts["graph_selected_cases"] += bool(g_ids)
        counts["graph_selected_instances"] += len(g_ids)
        counts["selected_more_than_configured_graph_slots_cases"] += len(g_ids) > configured_slots
        counts["selected_more_than_configured_graph_slots_instances"] += len(g_ids) if len(g_ids) > configured_slots else 0
        counts["selected_new_to_baseline_candidates"] += len(novel_ids)
        counts["selected_seen_in_baseline_candidates"] += len(g_ids) - len(novel_ids)
        counts["baseline_context_ids_displaced"] += len(lost)
        counts["same_ordered_ids"] += same_ids
        counts["same_ids_content_date"] += same_fingerprint
        counts["strict_exact_flips_with_same_context"] += same_fingerprint and left != right
        counts["selected_graph_records_with_session_date"] += sum(bool((graph_records[memory_id].get("metadata") or {}).get("session_date")) for memory_id in g_ids)
        counts["selected_graph_ids_found_as_vector_elsewhere"] += sum(memory_id in vector_by_id for memory_id in g_ids)
        counts["selected_graph_ids_with_recoverable_date"] += sum(bool((vector_by_id.get(memory_id, {}).get("metadata") or {}).get("session_date")) for memory_id in g_ids)
        counts["selected_graph_content_matching_vector_elsewhere"] += sum(memory_id in vector_by_id and graph_records[memory_id]["content"] == vector_by_id[memory_id]["content"] for memory_id in g_ids)
        for axis in (by_stratum[stratum], by_category[category][stratum]):
            axis["cases"] += 1
            axis[outcome] += 1
            axis["same_ids_content_date"] += same_fingerprint
            axis["strict_exact_flips_with_same_context"] += same_fingerprint and left != right
            axis["graph_selected_instances"] += len(g_ids)
            axis["baseline_ids_displaced"] += len(lost)
        cases.append({
            "case_id": case_id, "category": category, "stratum": stratum, "outcome": outcome,
            "selected_graph_ids": g_ids, "new_to_baseline_candidate_ids": novel_ids,
            "baseline_ids_displaced": lost, "graph_context_ids_gained": gained,
            "same_ordered_ids": same_ids, "same_ids_content_date": same_fingerprint,
        })
    report = {
        "scope": "Paired Chinese J cases: no-graph 2026-09-25 16:46 vs fused graph 17:21, both strict regrades.",
        "warning": "Graph selection is post-treatment. A graph-selected ID new to the baseline top15 is not necessarily a new fact, and per-stratum exact differences are not causal effects; answer generation is unstable even with identical context.",
        "counts": dict(counts),
        "graph_selected_count_distribution": {str(k): v for k, v in sorted(selection_distribution.items())},
        "by_stratum": {k: dict(v) for k, v in sorted(by_stratum.items())},
        "by_category": {k: {s: dict(c) for s, c in sorted(v.items())} for k, v in sorted(by_category.items())},
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("counts", "by_stratum", "by_category")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
