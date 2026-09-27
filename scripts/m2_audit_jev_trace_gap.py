#!/usr/bin/env python3
"""Inventory the archived Jev trace fields that the new diagnostic contract repairs."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "decider": ROOT / "results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl",
    "laya": ROOT / "results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl",
}
OUT = ROOT / "results/analysis/m2_jev_trace_gap_20260927.json"


def audit(path):
    counts = {"rows": 0, "scored": 0, "expanded": 0, "first_context_logged": 0,
              "first_state_logged": 0, "second_context_logged": 0,
              "expanded_graph_context_ids_stale": 0}
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        counts["rows"] += 1
        metadata = row.get("metadata") or {}
        info = metadata.get("jev_stop") or {}
        if not isinstance(info.get("first"), (float, int)):
            continue
        counts["scored"] += 1
        counts["first_context_logged"] += "first_context" in info
        counts["first_state_logged"] += "first_decision_state" in info
        if info.get("expanded"):
            counts["expanded"] += 1
            counts["second_context_logged"] += "second_context" in info
            graph_ids = {g.get("id") for g in row.get("graph_memories") or []}
            final_graph_ids = [mid for mid in metadata.get("answer_context_ids") or [] if mid in graph_ids]
            counts["expanded_graph_context_ids_stale"] += final_graph_ids != (metadata.get("graph_context_ids") or [])
    return counts


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    result = {
        "scope": "Historical Chinese Jev archive field coverage; read-only, zero model calls",
        "arms": {name: {"path": str(path.relative_to(ROOT)), **audit(path)} for name, path in ARMS.items()},
        "limits": ["These files precede the new trace contract and cannot reconstruct the missing first context", "Final graph membership is inferred by intersection of final answer IDs with archived final graph candidate IDs"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["arms"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
