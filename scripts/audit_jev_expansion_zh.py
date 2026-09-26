#!/usr/bin/env python3
"""Compare archived Jev expansion outputs with the same-night no-Jev pool.

The control is a proxy for the unlogged initial Jev pool, never a true
within-run pre-expansion snapshot. All strata are descriptive.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "results_zh/zhfull_fused_locomo_20260925_195248.jsonl"
CONTROL_GRADE = ROOT / "results/precision_regrade_fused.jsonl"
ARMS = {
    "decider": (
        ROOT / "results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl",
        ROOT / "results/precision_regrade_jevdec.jsonl",
    ),
    "laya": (
        ROOT / "results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl",
        ROOT / "results/precision_regrade_jevlaya.jsonl",
    ),
}
OUTPUT = ROOT / "results/analysis/jev_expansion_audit_20260926.json"
CATEGORIES = {"single-hop", "multi-hop", "temporal", "open-domain"}


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if row["case_id"] in rows:
            raise ValueError(f"duplicate case ID: {path}, {row['case_id']}")
        rows[row["case_id"]] = row
    return rows


def context(row: dict) -> list[tuple[str, str, str | None]]:
    records = {m["id"]: m for m in row.get("retrieved_memories", []) + row.get("graph_memories", [])}
    out = []
    for memory_id in row["metadata"]["answer_context_ids"]:
        record = records.get(memory_id)
        if record is None:
            raise ValueError(f"missing memory record: {row['case_id']}, {memory_id}")
        out.append((memory_id, record["content"], (record.get("metadata") or {}).get("session_date")))
    return out


def audit_arm(raw_path: Path, grade_path: Path, ctrl: dict, ctrl_grade: dict) -> dict:
    raw, grade = load(raw_path), load(grade_path)
    common = sorted(set(ctrl) & set(raw) & set(ctrl_grade) & set(grade))
    all_counts = Counter()
    strata: dict[str, Counter] = defaultdict(Counter)
    score_deltas: dict[str, list[float]] = defaultdict(list)
    cases = []
    for case_id in common:
        if ctrl_grade[case_id]["category"] not in CATEGORIES:
            continue
        left, right = ctrl[case_id], raw[case_id]
        info = right["metadata"].get("jev_stop") or {}
        if not info or not isinstance(info.get("first"), (int, float)):
            all_counts["paired_J_without_score"] += 1
            continue
        expanded = bool(info.get("expanded"))
        ctrl_retrieved = [m["id"] for m in left["retrieved_memories"]]
        ctrl_graph = [m["id"] for m in left["graph_memories"]]
        jev_retrieved = [m["id"] for m in right["retrieved_memories"]]
        jev_graph = [m["id"] for m in right["graph_memories"]]
        ctrl_pool = set(ctrl_retrieved + ctrl_graph)
        final_ids = right["metadata"]["answer_context_ids"]
        beyond_control_pool = [memory_id for memory_id in final_ids if memory_id not in ctrl_pool]
        stratum = "not_expanded" if not expanded else (
            "expanded_with_new_final_id_vs_control_pool" if beyond_control_pool else "expanded_without_new_final_id_vs_control_pool"
        )
        left_exact = ctrl_grade[case_id]["final"] == "exact"
        right_exact = grade[case_id]["final"] == "exact"
        outcome = "jev_only" if right_exact and not left_exact else "control_only" if left_exact and not right_exact else "same"
        same_context = context(left) == context(right)
        same_vector = ctrl_retrieved == jev_retrieved
        same_graph = ctrl_graph == jev_graph
        row = strata[stratum]
        row["cases"] += 1
        row[outcome] += 1
        row["same_full_context_vs_control"] += same_context
        row["same_vector_candidates_vs_control"] += same_vector
        row["same_graph_candidates_vs_control"] += same_graph
        row["control_graph_candidates"] += len(ctrl_graph)
        row["jev_final_graph_candidates"] += len(jev_graph)
        row["control_graph_candidates_in_jev_final"] += len(set(ctrl_graph) & set(jev_graph))
        row["all_control_graph_candidates_retained"] += set(ctrl_graph) <= set(jev_graph)
        row["final_ids_beyond_control_pool"] += len(beyond_control_pool)
        row["second_below_0_4"] += expanded and info.get("second", 1) < 0.4
        if expanded and isinstance(info.get("second"), (int, float)):
            score_deltas[stratum].append(info["second"] - info["first"])
        all_counts["paired_J_with_score"] += 1
        cases.append({
            "case_id": case_id, "category": ctrl_grade[case_id]["category"],
            "stratum": stratum, "outcome": outcome, "first": info["first"],
            "second": info.get("second"), "same_full_context_vs_control": same_context,
            "same_vector_candidates_vs_control": same_vector,
            "same_graph_candidates_vs_control": same_graph,
            "final_ids_beyond_control_pool": beyond_control_pool,
        })
    for stratum, deltas in score_deltas.items():
        strata[stratum]["second_minus_first_mean"] = round(mean(deltas), 4)
        strata[stratum]["same_score_to_0_01"] = sum(delta == 0 for delta in deltas)
    return {
        "raw_rows": len(raw), "grade_rows": len(grade), "paired_all": len(set(ctrl) & set(raw)),
        "counts": dict(all_counts),
        "strata": {key: dict(value) for key, value in sorted(strata.items())},
        "cases": cases,
    }


def main() -> None:
    ctrl, ctrl_grade = load(CONTROL), load(CONTROL_GRADE)
    report = {
        "method": "Paired Chinese strict J cases. Same-night no-Jev fused candidate pool is a proxy for unlogged Jev initial pool; compare Jev final selected IDs with that proxy pool.",
        "limits": "Separate runs can differ before expansion. New final ID vs control pool need not be caused by expansion; no-new final ID does not mean no candidates were fetched or context was unchanged. Strata are post-treatment and not causal effects.",
        "arms": {name: audit_arm(*paths, ctrl, ctrl_grade) for name, paths in ARMS.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {"counts": arm["counts"], "strata": arm["strata"]} for name, arm in report["arms"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
