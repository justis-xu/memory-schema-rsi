#!/usr/bin/env python3
"""Read-only audit of archived Chinese Jev score calibration and expansion.

The archived run does not record pre-expansion context IDs. AUC against answer
grades is therefore a proxy, not a ground-truth evidence-sufficiency test.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
CONTROL = "results/precision_regrade_fused.jsonl"
ARMS = {
    "decider": ("results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl", "results/precision_regrade_jevdec.jsonl"),
    "laya": ("results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl", "results/precision_regrade_jevlaya.jsonl"),
}
J_CATS = {"single-hop", "multi-hop", "temporal", "open-domain"}


def load(path: str) -> dict[str, dict]:
    return {
        row["case_id"]: row
        for line in (ROOT / path).read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def auc(values: list[tuple[float, bool]]) -> float | None:
    """Mann-Whitney AUC with half credit for score ties."""
    positives = sum(int(y) for _, y in values)
    negatives = len(values) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(values)
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(int(ordered[j][1]) for j in range(index, end))
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def conv(case_id: str) -> str:
    match = re.search(r"conv-\d+", case_id)
    assert match, case_id
    return match.group(0)


def auc_ci(rows: list[dict], key: str, draws: int = 4000) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[conv(row["case_id"])].append(row)
    names = sorted(groups)
    rng = random.Random(20260926)
    values = []
    for _ in range(draws):
        sampled = [item for _ in names for item in groups[rng.choice(names)]]
        estimate = auc([(row["first"], row[key]) for row in sampled])
        if estimate is not None:
            values.append(estimate)
    values.sort()
    return {
        "point": round(auc([(row["first"], row[key]) for row in rows]) or 0, 4),
        "conversation_bootstrap_95": [round(values[int(len(values) * 0.025)], 4), round(values[int(len(values) * 0.975)], 4)],
        "draws": draws, "seed": 20260926,
    }


def summarize(name: str, raw_path: str, grade_path: str, control: dict) -> dict:
    raw, grade = load(raw_path), load(grade_path)
    infos = [row.get("metadata", {}).get("jev_stop") for row in raw.values()]
    valid = [info for info in infos if info and isinstance(info.get("first"), (int, float))]
    expanded = [info for info in valid if info.get("expanded") and isinstance(info.get("second"), (int, float))]
    deltas = [info["second"] - info["first"] for info in expanded]
    initial_context_records = []
    literal_probe = Counter()
    hidden_literal_examples = []
    for row in raw.values():
        info = row.get("metadata", {}).get("jev_stop")
        if not info or info.get("expanded"):
            continue  # for these rows the saved final context is also the first-scored context
        records = {record["id"]: record for record in row.get("retrieved_memories", []) + row.get("graph_memories", [])}
        context = [records[mem_id] for mem_id in row["metadata"]["answer_context_ids"] if mem_id in records]
        initial_context_records.extend(context)
        if row["metadata"].get("category") in J_CATS:
            gold = str(row.get("expected_answer") or "")
            normalized = lambda value: re.sub(r"[\W_]+", "", value.lower())  # noqa: E731
            if len(normalized(gold)) >= 3 and not re.search(r"[,，、;/；]", gold):
                literal_probe["eligible"] += 1
                full = normalized(" ".join(record.get("content") or "" for record in context))
                digest = normalized(" | ".join((record.get("content") or "")[:60] for record in context[:15])[:950])
                in_full, in_digest = normalized(gold) in full, normalized(gold) in digest
                literal_probe["gold_literal_in_full_content"] += in_full
                literal_probe["gold_literal_in_digest"] += in_digest
                literal_probe["full_but_not_digest"] += in_full and not in_digest
                if in_full and not in_digest:
                    hidden_literal_examples.append({
                        "case_id": row["case_id"], "category": row["metadata"]["category"],
                        "question": row["question"], "gold": gold, "first_score": info["first"],
                        "strict_grade": grade.get(row["case_id"], {}).get("final"),
                    })
    pairs = []
    for case_id in sorted(set(raw) & set(grade) & set(control)):
        info = raw[case_id].get("metadata", {}).get("jev_stop")
        if not info or not isinstance(info.get("first"), (int, float)) or grade[case_id]["category"] not in J_CATS:
            continue
        pairs.append({
            "case_id": case_id, "category": grade[case_id]["category"],
            "first": float(info["first"]), "expanded": bool(info.get("expanded")),
            "control_exact": control[case_id]["final"] == "exact",
            "arm_exact": grade[case_id]["final"] == "exact",
        })
    per_category = {}
    for category in sorted(J_CATS):
        subset = [row for row in pairs if row["category"] == category]
        per_category[category] = {
            "cases": len(subset), "expanded": sum(row["expanded"] for row in subset),
            "auc_first_vs_control_exact": round(auc([(row["first"], row["control_exact"]) for row in subset]) or 0, 4),
        }
    by_trigger = {}
    for label, trigger in (("expanded", True), ("not_expanded", False)):
        subset = [row for row in pairs if row["expanded"] == trigger]
        by_trigger[label] = {
            "cases": len(subset), "control_exact": sum(row["control_exact"] for row in subset),
            "jev_exact": sum(row["arm_exact"] for row in subset),
        }
    return {
        "raw_rows": len(raw), "grade_rows": len(grade), "missing_jev_metadata": len(infos) - len(valid),
        "first_scores": len(valid), "first_mean": round(mean(info["first"] for info in valid), 4),
        "threshold_counts_from_rounded_first": {str(threshold): sum(info["first"] < threshold for info in valid) for threshold in (0.3, 0.4, 0.5)},
        "expanded": len(expanded), "second_below_0_4": sum(info["second"] < 0.4 for info in expanded),
        "second_minus_first": {
            "mean": round(mean(deltas), 4), "median": round(median(deltas), 4),
            "positive": sum(delta > 0 for delta in deltas), "zero_at_0_01_precision": sum(delta == 0 for delta in deltas),
            "negative": sum(delta < 0 for delta in deltas),
        },
        "known_initial_context_visibility": {
            "records": len(initial_context_records),
            "content_longer_than_60_characters": sum(len(record.get("content") or "") > 60 for record in initial_context_records),
            "records_with_session_date_metadata": sum(bool((record.get("metadata") or {}).get("session_date")) for record in initial_context_records),
            "note": "The decision digest includes only the first 60 content characters per memory and never includes session_date metadata; the answer prompt sees full content and dates.",
        },
        "gold_literal_probe_nonexpanded_J": dict(literal_probe),
        "gold_literal_hidden_examples": hidden_literal_examples,
        "paired_J": len(pairs), "auc_first_vs_control_exact": auc_ci(pairs, "control_exact"),
        "auc_first_vs_jev_exact": auc_ci(pairs, "arm_exact"),
        "by_trigger": by_trigger, "per_category": per_category,
    }


def main() -> None:
    control = load(CONTROL)
    report = {
        "method": "Archived Jev first/second probabilities; paired strict exact as a proxy; conversation-cluster AUC bootstrap.",
        "limits": "First score was computed on an unrecorded pre-expansion context digest. Control and Jev retrieval can differ. AUC against answer correctness is not direct evidence-sufficiency calibration. Scores are rounded to 0.01; threshold replays are call-count predictions only.",
        "arms": {name: summarize(name, *paths, control) for name, paths in ARMS.items()},
    }
    output = ROOT / "results/analysis/jev_score_audit_20260926.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "arms": report["arms"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
