#!/usr/bin/env python3
"""Exact no-expansion Jev state-length audit from archived final context IDs."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "decider": ROOT / "results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl",
    "laya": ROOT / "results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl",
}
OUT = ROOT / "results/analysis/m2_jev_state_cap_20260927.json"


def audit(path):
    rows, examples = 0, []
    aggregate = {"scored_unexpanded": 0, "mapped": 0, "unmapped": 0,
                 "digest_over_950": 0, "state_over_1000_before_cap": 0,
                 "any_memory_text_cut_by_outer_caps": 0,
                 "max_digest_chars": 0, "max_state_chars_before_cap": 0,
                 "total_context_memories": 0}
    for line in path.open():
        row = json.loads(line)
        rows += 1
        info = (row.get("metadata") or {}).get("jev_stop") or {}
        if not isinstance(info.get("first"), (int, float)) or info.get("expanded"):
            continue
        aggregate["scored_unexpanded"] += 1
        ids = row["metadata"]["answer_context_ids"][:15]
        contents = {m["id"]: m["content"] for m in row["retrieved_memories"]
                    + row.get("graph_memories", [])}
        if any(mid not in contents for mid in ids):
            aggregate["unmapped"] += 1
            continue
        aggregate["mapped"] += 1
        aggregate["total_context_memories"] += len(ids)
        digest = " | ".join(contents[mid][:60] for mid in ids)
        state = f"Question: {row['question']}\n\nContext:\n{digest[:950]}"
        aggregate["digest_over_950"] += len(digest) > 950
        aggregate["state_over_1000_before_cap"] += len(state) > 1000
        aggregate["any_memory_text_cut_by_outer_caps"] += len(digest) > 950 or len(state) > 1000
        aggregate["max_digest_chars"] = max(aggregate["max_digest_chars"], len(digest))
        aggregate["max_state_chars_before_cap"] = max(aggregate["max_state_chars_before_cap"], len(state))
        if len(examples) < 3:
            examples.append({"case_id": row["case_id"], "context_memories": len(ids),
                             "digest_chars": len(digest), "state_chars": len(state)})
    aggregate["archive_rows"] = rows
    aggregate["first_examples"] = examples
    assert aggregate["mapped"] == aggregate["scored_unexpanded"]
    return aggregate


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    result = {"scope": "Read-only exact Jev state-length reconstruction for scored non-expanded Chinese historical cases; no model calls",
              "arms": {name: {"path": str(path.relative_to(ROOT)), **audit(path)}
                       for name, path in ARMS.items()},
              "limits": ["Expanded cases lack first ordered context, so their first states cannot be reconstructed",
                         "This audits the 950/1000 character outer caps, not the already known 60-character per-memory loss",
                         "No inference about Jev decision accuracy follows from input length alone"]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print({k: {x: v for x, v in a.items() if x not in ("first_examples", "path")}
           for k, a in result["arms"].items()})


if __name__ == "__main__":
    main()
