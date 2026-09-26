#!/usr/bin/env python3
"""Check structural integrity of English/Chinese LoCoMo gold evidence IDs.

This does not establish semantic relevance of a valid evidence turn.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "data/locomo/locomo10.json"
ZH = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUTPUT = ROOT / "results/analysis/gold_evidence_id_audit_20260926.json"
SNAPSHOTS = ROOT / "experiments/schema_rsi/runs"
ID = re.compile(r"D:?(\d+):(\d+)")


def normalize(value: re.Match[str]) -> str:
    return f"D{int(value.group(1))}:{int(value.group(2))}"


def main() -> None:
    en = {entry["sample_id"]: entry for entry in json.loads(EN.read_text())}
    zh = {entry["sample_id"]: entry for entry in json.loads(ZH.read_text())}
    assert en.keys() == zh.keys()
    counts = Counter()
    anomalies = []
    for conv, entry in en.items():
        translated = zh[conv]
        assert len(entry["qa"]) == len(translated["qa"])
        en_turns = {turn["dia_id"] for key, value in entry["conversation"].items() if re.fullmatch(r"session_\d+", key) for turn in value}
        zh_turns = {turn["dia_id"] for key, value in translated["conversation"].items() if re.fullmatch(r"session_\d+", key) for turn in value}
        assert en_turns == zh_turns
        for index, (source, target) in enumerate(zip(entry["qa"], translated["qa"])):
            counts["qa"] += 1
            case_id = f"locomo_{conv}_qa{index}"
            if source.get("evidence") != target.get("evidence"):
                counts["changed_evidence_between_languages"] += 1
                anomalies.append({"case_id": case_id, "kind": "changed_between_languages", "en": source.get("evidence"), "zh": target.get("evidence")})
            evidence = source.get("evidence") or []
            counts["empty_evidence_qa"] += not evidence
            counts["evidence_list_items"] += len(evidence)
            for item in evidence:
                matches = list(ID.finditer(str(item)))
                if not matches:
                    counts["unparseable_items"] += 1
                    anomalies.append({"case_id": case_id, "kind": "unparseable", "raw": item})
                    continue
                if len(matches) > 1:
                    counts["multiple_ids_in_one_item"] += 1
                    anomalies.append({"case_id": case_id, "kind": "multiple_ids_in_one_item", "raw": item, "normalized": [normalize(m) for m in matches]})
                for match in matches:
                    counts["parsed_references"] += 1
                    canonical = normalize(match)
                    raw = match.group()
                    if raw != canonical:
                        counts["noncanonical_references"] += 1
                        anomalies.append({"case_id": case_id, "kind": "noncanonical", "raw": raw, "normalized": canonical})
                    if canonical not in en_turns:
                        counts["missing_references"] += 1
                        anomalies.append({"case_id": case_id, "kind": "missing_turn", "raw": raw, "normalized": canonical})
    snapshot_counts = Counter()
    unreachable = []
    for path in sorted(SNAPSHOTS.glob("conv-*_full_frozen.json")):
        snapshot = json.loads(path.read_text())
        source_turns = {turn["dia_id"] for session in snapshot["sessions"] for turn in session["turns"]}
        for case in snapshot["cases"]:
            if case["category"] == "adversarial" or not case.get("evidence"):
                continue
            snapshot_counts["scored_cases"] += 1
            absent = [e for e in case["evidence"] if e not in source_turns]
            if absent:
                snapshot_counts["unreachable_cases"] += 1
                snapshot_counts["unreachable_raw_items"] += len(absent)
                unreachable.append({"case_id": case["id"], "unreachable_raw_items": absent})
    report = {
        "scope": "All 1,986 aligned English/Chinese LoCoMo QA evidence lists and source turn IDs.",
        "warning": "A structurally valid evidence ID can still point to an irrelevant turn or support the wrong relation; this script cannot detect those semantic errors.",
        "counts": dict(counts), "anomalies": anomalies,
        "rsi_frozen_snapshots": {"counts": dict(snapshot_counts), "unreachable_cases": unreachable},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
