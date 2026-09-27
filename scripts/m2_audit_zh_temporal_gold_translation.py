#!/usr/bin/env python3
"""Screen unambiguous English/Chinese gold month translations and archived impact."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh")
INVENTORY = ROOT / "results/analysis/m2_historical_run_inventory_20260926.json"
OUT = ROOT / "results/analysis/m2_zh_temporal_gold_translation_20260927.json"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
CHINESE = {s: i for i, s in enumerate("一二三四五六七八九十", 1)}
CHINESE.update({"十一": 11, "十二": 12})


def english_months(answer):
    return {i for i, month in enumerate(MONTHS, 1)
            if re.search(r"\b" + month + r"\b", str(answer), re.I)}


def chinese_months(answer):
    answer = str(answer)
    found = {int(v) for v in re.findall(r"(?<!\d)(1[0-2]|[1-9])月", answer)}
    found.update(CHINESE[v] for v in re.findall(r"([一二三四五六七八九十]+)月", answer)
                 if v in CHINESE)
    return found


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    english = {s["sample_id"]: s for s in json.loads((DATA / "locomo10.json").read_text())}
    chinese = {s["sample_id"]: s for s in json.loads((DATA / "locomo10_zh.json").read_text())}
    assert english.keys() == chinese.keys()
    comparable, mismatches = [], []
    for conversation in sorted(english):
        en, zh = english[conversation], chinese[conversation]
        assert len(en["qa"]) == len(zh["qa"])
        for index, (a, b) in enumerate(zip(en["qa"], zh["qa"])):
            en_month, zh_month = english_months(a.get("answer", "")), chinese_months(b.get("answer", ""))
            if len(en_month) != 1 or len(zh_month) != 1:
                continue
            cid = f"locomo_{conversation}_qa{index}"
            comparable.append(cid)
            if en_month != zh_month:
                mismatches.append({"case_id": cid, "category": a["category"],
                                   "english": {"question": a["question"], "gold": a["answer"],
                                               "month": next(iter(en_month))},
                                   "chinese": {"question": b["question"], "gold": b["answer"],
                                               "month": next(iter(zh_month))},
                                   "evidence_ids": a["evidence"]})
    assert len(comparable) == 206 and len(mismatches) == 1
    hit = mismatches[0]
    assert hit["case_id"] == "locomo_conv-30_qa15"
    sample = chinese["conv-30"]
    evidence = []
    for tid in hit["evidence_ids"]:
        day, number = tid.split(":")
        key = "session_" + day[1:]
        turn = sample["conversation"][key][int(number) - 1]
        assert turn["dia_id"] == tid
        evidence.append({"turn_id": tid, "session_date": sample["conversation"][key + "_date_time"],
                         "speaker": turn["speaker"], "text": turn["text"]})
    hit["chinese_source_evidence"] = evidence
    inventory = json.loads(INVENTORY.read_text())
    exposures = []
    for item in inventory["chinese_locomo_runs"]:
        path = ROOT / item["path"]
        records = [row for line in path.open() if (row := json.loads(line))
                   and row["case_id"] == hit["case_id"]]
        assert len(records) == 1
        row = records[0]
        exposures.append({"path": item["path"], "expected": row["expected_answer"],
                          "prediction": row["predicted_answer"],
                          "judge_correct": row.get("metrics", {}).get("judge_correct")})
    result = {"scope": "Read-only literal month comparison in aligned LoCoMo English/Chinese gold answers; no model calls",
              "total_qa": sum(len(s["qa"]) for s in english.values()),
              "single_month_comparable": len(comparable), "mismatch_count": len(mismatches),
              "mismatches": mismatches, "eight_chinese_main_run_exposure": exposures,
              "limits": ["Only gold answers with exactly one full English month name and one Chinese month expression were compared",
                         "The other 1,780 QA were not ruled free of translation errors",
                         "Historical judge outcomes are observational; correcting the dataset would require a separate versioned evaluation"]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(len(comparable), "comparable,", len(mismatches), "mismatch,", len(exposures), "run records")


if __name__ == "__main__":
    main()
