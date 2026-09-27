#!/usr/bin/env python3
"""Screen archived Chinese temporal misses before spending answer calls on date order."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
REGRADE = ROOT / "results/precision_regrade_zhfull.jsonl"
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_temporal_order_candidate_screen_20260927.json"
DATE = re.compile(r"20\d{2}年\d{1,2}月(?:\d{1,2}日)?")
MONTH = re.compile(r"(20\d{2}年\d{1,2}月)")
CASE_STUDIES = ["conv-50_qa12", "conv-50_qa16", "conv-43_qa23",
                "conv-42_qa22", "conv-48_qa71"]


def rows(path):
    return {row["case_id"]: row for line in path.open() if (row := json.loads(line))}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    raw, graded = rows(RAW), rows(REGRADE)
    aligned = [(raw[k], graded[k]) for k in raw.keys() & graded.keys()
               if raw[k]["predicted_answer"] == graded[k]["pred"]]
    temporal = [(r, g) for r, g in aligned if r["metadata"]["category"] == "temporal"]
    nonexact = [(r, g) for r, g in temporal if g["final"] != "exact"]
    dated = [(r, g) for r, g in nonexact if MONTH.search(g["gold"])]
    broad, strict = [], []
    for r, g in dated:
        context = " ".join(m["content"] for m in r["retrieved_memories"][:15])
        dates = set(DATE.findall(context))
        gold_month = MONTH.search(g["gold"]).group(1)
        if gold_month in context and len(dates) >= 2:
            broad.append(r["case_id"])
        gold_literal = DATE.search(g["gold"]).group()
        if gold_literal in dates and len(dates) >= 2:
            strict.append(r["case_id"])
    assert len(temporal) == 321 and len(nonexact) == 149 and len(dated) == 107
    assert len(broad) == 80 and len(strict) == 35

    samples = {s["sample_id"]: s for s in json.loads(DATASET.read_text())}
    studies = []
    for short in CASE_STUDIES:
        case_id = "locomo_" + short
        r, g = raw[case_id], graded[case_id]
        conv, q = short.split("_qa")
        sample = samples[conv]
        qa = sample["qa"][int(q)]
        evidence = []
        turn_ids = list(qa["evidence"])
        if short == "conv-48_qa71":
            turn_ids.append("D24:3")  # same session, explicit community gathering
        for turn_id in turn_ids:
            day, number = turn_id.split(":")
            key = "session_" + day[1:]
            turn = sample["conversation"][key][int(number) - 1]
            assert turn["dia_id"] == turn_id
            evidence.append({"turn_id": turn_id, "listed_as_gold_evidence": turn_id in qa["evidence"],
                             "session_date": sample["conversation"][key + "_date_time"],
                             "speaker": turn["speaker"], "text": turn["text"]})
        studies.append({"case_id": case_id, "question": qa["question"],
                        "gold": qa["answer"], "prediction": g["pred"],
                        "regrade": g["final"], "evidence": evidence,
                        "top_memories": [{"id": m["id"], "content": m["content"],
                                          "session_date": m["metadata"].get("session_date")}
                                         for m in r["retrieved_memories"][:3]]})
    result = {
        "scope": "Matched archived Chinese no-graph run and precision regrade; read-only, no model calls",
        "inputs": [str(RAW.relative_to(ROOT)), str(REGRADE.relative_to(ROOT)), str(DATASET)],
        "counts": {"raw": len(raw), "graded": len(graded), "matched_prediction": len(aligned),
                   "temporal": len(temporal), "nonexact": len(nonexact),
                   "gold_has_year_month": len(dated),
                   "gold_month_in_top15_and_multiple_date_literals": len(broad),
                   "strict_gold_literal_in_top15_and_multiple_date_literals": len(strict)},
        "broad_case_ids": sorted(broad), "strict_case_ids": sorted(strict),
        "selected_source_packets": studies,
        "limits": ["Date literal matching does not establish that the memory concerns the asked event",
                   "Five source packets were selected for mechanism diversity, not random sampling",
                   "No date-order counterfactual or answer calls were run"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["counts"])


if __name__ == "__main__":
    main()
