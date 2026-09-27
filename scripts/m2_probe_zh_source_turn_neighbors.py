#!/usr/bin/env python3
"""Read-only adjacent-turn expansion of frozen Chinese lexical source candidates."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/analysis/m2_zh_source_tokenizer_comparison_20260927.json"
DATA = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_zh_source_turn_neighbors_20260927.json"
K = 15


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    baseline = json.loads(BASE.read_text())
    dataset = {s["sample_id"]: s for s in json.loads(DATA.read_text())}
    by_id = {row["case_id"]: row for row in baseline["by_case"]}
    rows = []
    for item in baseline["audited_20"]:
        cid = item["case_id"]
        conv = cid.removeprefix("locomo_").split("_qa")[0]
        seeds = by_id[cid]["cjk_bigram"]["top15"]
        selected, reasons = [], {}
        for seed in seeds:
            tid = seed["turn_id"]
            day, number = tid.split(":")
            turns = dataset[conv]["conversation"]["session_" + day[1:]]
            index = int(number) - 1
            for near, reason in ((index, "lexical_seed"), (index - 1, "previous_turn"),
                                 (index + 1, "next_turn")):
                if not 0 <= near < len(turns) or len(selected) >= K:
                    continue
                turn_id = turns[near]["dia_id"]
                if turn_id not in reasons:
                    selected.append(turn_id)
                    reasons[turn_id] = reason
        audited = set(item["audited_turn_ids"])
        seed_ids = [s["turn_id"] for s in seeds]
        rows.append({"case_id": cid, "label": item["label"],
                     "baseline_ids": seed_ids, "expanded_ids": selected,
                     "expanded_reasons": reasons,
                     "audited_ids": sorted(audited),
                     "baseline_audited": sorted(audited & set(seed_ids)),
                     "expanded_audited": sorted(audited & set(selected)),
                     "added_audited": sorted(audited & (set(selected) - set(seed_ids))),
                     "displaced_seeds": sorted(set(seed_ids) - set(selected))})
    def summary(key):
        return {"cases_with_any_audited_id": sum(bool(r[key]) for r in rows),
                "audited_id_exposures": sum(len(r[key]) for r in rows)}
    result = {"scope": "Fixed 20 previously source-audited Chinese QA; reuse frozen CJK bigram lexical seed rankings and same 15 source-turn slots; no model calls",
              "policy": "For each ranked lexical seed, append the seed then its previous and next turn from the same session, deduplicate, stop at 15; no gold/source labels used in selection",
              "baseline": summary("baseline_audited"),
              "expanded": summary("expanded_audited"),
              "cases_with_displaced_seeds": sum(bool(r["displaced_seeds"]) for r in rows),
              "total_source_slots_baseline": sum(len(r["baseline_ids"]) for r in rows),
              "total_source_slots_expanded": sum(len(r["expanded_ids"]) for r in rows),
              "rows": rows,
              "limits": ["Audited turn IDs are incomplete diagnostic references, not a full support gold",
                         "An adjacent turn may expose a wrong-person fact or adversarial answer",
                         "This is source-turn slot exposure, not Mem0 retrieval, Graph/Jev benefit or QA accuracy"]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["baseline"], result["expanded"],
          "slots", result["total_source_slots_baseline"], result["total_source_slots_expanded"])


if __name__ == "__main__":
    main()
