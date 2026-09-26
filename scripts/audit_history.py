#!/usr/bin/env python3
"""Read-only, reproducible audit of archived LoCoMo runs.

Uses paired case IDs and conversation-level resampling. It does not call models,
touch the vector store, or mutate experiment results.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J_CATS = {"single-hop", "multi-hop", "temporal", "open-domain"}
ARMS = {
    "zh_baseline_repeat": ("results/precision_regrade_zhfull.jsonl", "results/precision_regrade_base.jsonl"),
    "zh_graph_repeat": ("results/precision_regrade_graph.jsonl", "results/precision_regrade_fused.jsonl"),
    "zh_graph": ("results/precision_regrade_base.jsonl", "results/precision_regrade_graph.jsonl"),
    "zh_jevdec": ("results/precision_regrade_fused.jsonl", "results/precision_regrade_jevdec.jsonl"),
    "zh_jevlaya": ("results/precision_regrade_fused.jsonl", "results/precision_regrade_jevlaya.jsonl"),
    "en_final": ("results/precision_regrade_C.jsonl", "results/precision_regrade_D.jsonl"),
}
RAW = {
    "zh_baseline_repeat": ("results_zh/zhfull_locomo_20260925_115806.jsonl", "results_zh/zhfull_locomo_20260925_164645.jsonl"),
    "zh_graph_repeat": ("results_zh/zhfull_fused_locomo_20260925_172126.jsonl", "results_zh/zhfull_fused_locomo_20260925_195248.jsonl"),
    "zh_graph": ("results_zh/zhfull_locomo_20260925_164645.jsonl", "results_zh/zhfull_fused_locomo_20260925_172126.jsonl"),
    "zh_jevdec": ("results_zh/zhfull_fused_locomo_20260925_195248.jsonl", "results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl"),
    "zh_jevlaya": ("results_zh/zhfull_fused_locomo_20260925_195248.jsonl", "results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl"),
    "en_final": ("results/full_locomo_finalC_ctrl_20260925_090746.jsonl", "results/full_locomo_finalD_champ_20260925_093052.jsonl"),
}


def read_jsonl(path: str) -> dict[str, dict]:
    rows = {}
    for line in (ROOT / path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row["case_id"]
        if case_id in rows:
            raise ValueError(f"duplicate case ID in {path}: {case_id}")
        rows[case_id] = row
    return rows


def conversation(case_id: str) -> str:
    match = re.search(r"locomo_(conv-\d+)_qa\d+", case_id)
    if not match:
        raise ValueError(case_id)
    return match.group(1)


def context_fingerprint(row: dict) -> list[tuple]:
    records = {m["id"]: m for m in row.get("retrieved_memories", []) + row.get("graph_memories", [])}
    out = []
    for mem_id in row.get("metadata", {}).get("answer_context_ids", []):
        record = records.get(mem_id)
        if record is None:
            out.append((mem_id, None, None))
        else:
            out.append((mem_id, record.get("content"), (record.get("metadata") or {}).get("session_date")))
    return out


def audit_pair(name: str, draws: int, seed: int) -> dict:
    left, right = (read_jsonl(p) for p in ARMS[name])
    left_raw, right_raw = (read_jsonl(p) for p in RAW[name])
    common = set(left) & set(right)
    cases = [case_id for case_id in sorted(common) if left[case_id]["category"] in J_CATS]
    if any(left[c]["category"] != right[c]["category"] for c in cases):
        raise ValueError(f"category mismatch: {name}")
    counts = Counter()
    by_category = defaultdict(Counter)
    by_conversation = defaultdict(Counter)
    per_conv = defaultdict(list)
    for case_id in cases:
        a, b = left[case_id]["final"] == "exact", right[case_id]["final"] == "exact"
        label = "both" if a and b else "left_only" if a else "right_only" if b else "neither"
        counts[label] += 1
        by_category[left[case_id]["category"]][label] += 1
        conv = conversation(case_id)
        by_conversation[conv][label] += 1
        per_conv[conv].append((int(b) - int(a), 1))
    rng = random.Random(seed)
    convs = sorted(per_conv)
    summed = {conv: (sum(d for d, _ in per_conv[conv]), len(per_conv[conv])) for conv in convs}
    samples = []
    for _ in range(draws):
        picked = [summed[rng.choice(convs)] for _ in convs]
        samples.append(sum(d for d, _ in picked) / sum(n for _, n in picked))
    samples.sort()
    raw_common = sorted(set(left_raw) & set(right_raw) & set(cases))
    same_ids = same_fingerprint = changed_answer_same = flipped_same = 0
    for case_id in raw_common:
        a, b = left_raw[case_id], right_raw[case_id]
        if a.get("metadata", {}).get("answer_context_ids") != b.get("metadata", {}).get("answer_context_ids"):
            continue
        same_ids += 1
        if context_fingerprint(a) != context_fingerprint(b):
            continue
        same_fingerprint += 1
        changed_answer_same += a.get("predicted_answer") != b.get("predicted_answer")
        flipped_same += (left[case_id]["final"] == "exact") != (right[case_id]["final"] == "exact")
    return {
        "left": ARMS[name][0], "right": ARMS[name][1],
        "left_rows": len(left), "right_rows": len(right), "paired_all": len(common),
        "paired_J": len(cases), "paired_case_ids": cases,
        "outcomes": dict(counts),
        "delta_exact": counts["right_only"] - counts["left_only"],
        "delta_pp": round(100 * (counts["right_only"] - counts["left_only"]) / len(cases), 3),
        "by_category": {key: dict(value) for key, value in sorted(by_category.items())},
        "by_conversation": {key: dict(value) for key, value in sorted(by_conversation.items())},
        "cluster_bootstrap": {
            "unit": "conversation", "draws": draws, "seed": seed,
            "delta_pp_95_interval": [round(100 * samples[int(draws * 0.025)], 3), round(100 * samples[int(draws * 0.975)], 3)],
            "fraction_nonpositive": round(sum(x <= 0 for x in samples) / draws, 4),
        },
        "same_context": {
            "same_ordered_ids": same_ids, "same_ids_content_and_date": same_fingerprint,
            "different_answer": changed_answer_same, "strict_exact_flips": flipped_same,
        },
    }


def audit_rounds() -> dict:
    rows = [json.loads(line) for line in (ROOT / "results/search_rounds.jsonl").read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    details = {p.stem: sum(1 for line in p.open() if line.strip()) for p in (ROOT / "data/search/rounds").glob("*.jsonl")}
    return {
        "summary_count": len(rows), "unique_summary_ids": len(set(ids)),
        "summary_missing_detail": sorted(set(ids) - set(details)),
        "detail_without_summary": {key: value for key, value in sorted(details.items()) if key not in ids},
        "nonstandard_detail_rows": {key: value for key, value in sorted(details.items()) if key in ids and value != 616},
        "calibration_exact": {r["id"]: r["exact"] for r in rows if "calib" in r["id"]},
        "selected_repeats_exact": {r["id"]: r["exact"] for r in rows if any(token in r["id"] for token in ("rep", "noatomic", "atomic", "jev_stop"))},
    }


def audit_chinese_sources() -> dict:
    path = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        ids = {row[0] for row in db.execute("SELECT embedding_id FROM embeddings")}
    cache = {}
    for name in ("structured_cache.json", "topic_cache.json", "memory_embeddings.json"):
        keys = set(json.loads((ROOT / "data/graph_zh" / name).read_text()))
        cache[name] = {"keys": len(keys), "current_id_overlap": len(keys & ids), "stale_keys": len(keys - ids), "current_ids_missing": len(ids - keys)}
    locomo = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
    data = json.loads(locomo.read_text()) if locomo.exists() else []
    turns = [turn for entry in data for key, value in entry["conversation"].items() if re.fullmatch(r"session_\d+", key) and isinstance(value, list) for turn in value]
    return {
        "chroma_embedding_ids": len(ids), "graph_cache": cache,
        "translated_locomo_exists": locomo.exists(), "conversations": len(data),
        "questions": sum(len(entry["qa"]) for entry in data), "turns": len(turns),
        "turns_with_english_image_caption": sum(bool(turn.get("blip_caption")) for turn in turns),
    }


def audit_chinese_evidence_difficulty() -> dict:
    path = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text())
    grades = read_jsonl("results/precision_regrade_zhfull.jsonl")
    counts = defaultdict(Counter)
    for entry in data:
        for index, qa in enumerate(entry["qa"]):
            case_id = f"locomo_{entry['sample_id']}_qa{index}"
            row = grades.get(case_id)
            if row is None or row["category"] not in J_CATS:
                continue
            evidence = qa.get("evidence") or []
            sessions = {match.group(1) for item in evidence if (match := re.match(r"D(\d+):", str(item)))}
            for axis, size in (("turns", len(evidence)), ("sessions", len(sessions))):
                bucket = str(size) if size < (4 if axis == "turns" else 3) else ("4+" if axis == "turns" else "3+")
                key = (row["category"], axis, bucket)
                counts[key]["cases"] += 1
                counts[key]["exact"] += row["final"] == "exact"
    return {"available": True, "source_grade": "precision_regrade_zhfull", "groups": {"|".join(key): dict(value) for key, value in sorted(counts.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--output", type=Path, default=ROOT / "results/analysis/history_audit_20260926.json")
    args = ap.parse_args()
    result = {
        "method": "Archived strict grades paired by case_id; conversation cluster bootstrap for uncertainty. Same-context comparison uses archived memory contents and session dates.",
        "limits": "Bootstrap has only ten conversation clusters. Same-context answer changes are descriptive, not an estimate of model variance under controlled repeats. English final D is a multi-component stack. Chinese records predate store repair.",
        "chinese_sources": audit_chinese_sources(),
        "chinese_evidence_difficulty": audit_chinese_evidence_difficulty(),
        "rounds": audit_rounds(),
        "pairs": {name: audit_pair(name, args.draws, 20260926) for name in ARMS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rounds": result["rounds"], "pairs": {key: {k: v for k, v in pair.items() if k in ("paired_J", "outcomes", "delta_exact", "delta_pp", "cluster_bootstrap", "same_context")} for key, pair in result["pairs"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
