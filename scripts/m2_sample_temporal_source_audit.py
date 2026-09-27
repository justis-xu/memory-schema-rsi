#!/usr/bin/env python3
"""Reproducible source packet for a small random Chinese temporal nonexact sample."""

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
REGRADE = ROOT / "results/precision_regrade_zhfull.jsonl"
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_temporal_source_sample_20260927.json"
SEED = 20260927
N = 12


def keyed(path):
    return {item["case_id"]: item for line in path.open() if (item := json.loads(line))}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    raw, graded = keyed(RAW), keyed(REGRADE)
    eligible = sorted(cid for cid in raw.keys() & graded.keys()
                      if raw[cid]["predicted_answer"] == graded[cid]["pred"]
                      and raw[cid]["metadata"]["category"] == "temporal"
                      and graded[cid]["final"] != "exact")
    assert len(eligible) == 149
    sample_ids = random.Random(SEED).sample(eligible, N)
    samples = {s["sample_id"]: s for s in json.loads(DATASET.read_text())}
    cases = []
    for cid in sample_ids:
        short = cid.removeprefix("locomo_")
        conversation, index = short.split("_qa")
        sample = samples[conversation]
        qa = sample["qa"][int(index)]
        source = []
        for tid in qa.get("evidence", []):
            day, number = tid.split(":")
            key = "session_" + day[1:]
            turn = sample["conversation"][key][int(number) - 1]
            assert turn["dia_id"] == tid
            source.append({"turn_id": tid,
                           "session_date": sample["conversation"][key + "_date_time"],
                           "speaker": turn["speaker"], "text": turn["text"],
                           "caption": turn.get("blip_caption"), "query": turn.get("query")})
        cases.append({"case_id": cid, "question": qa["question"], "gold": qa["answer"],
                      "pred": graded[cid]["pred"], "grade": graded[cid]["final"],
                      "evidence": source,
                      "top3": [{"memory_id": m["id"], "content": m["content"],
                                "session_date": m["metadata"].get("session_date")}
                               for m in raw[cid]["retrieved_memories"][:3]]})
    result = {"scope": "Random fixed sample of matched archived Chinese temporal nonexact cases; source packets only, no model calls",
              "seed": SEED, "sample_size": N, "eligible_size": len(eligible),
              "inputs": [str(RAW.relative_to(ROOT)), str(REGRADE.relative_to(ROOT)), str(DATASET)],
              "sampled_ids": sample_ids, "cases": cases,
              "limits": ["Benchmark-listed evidence may omit relevant neighboring turns or identify the wrong event",
                         "The packet contains source text, not automatic semantic judgments",
                         "Twelve cases do not establish a population error rate"]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("eligible", len(eligible), "sample", sample_ids)


if __name__ == "__main__":
    main()
