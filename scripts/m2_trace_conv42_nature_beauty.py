#!/usr/bin/env python3
"""Trace one Chinese source-backed joint fact through current memories and frozen retrieval."""

import hashlib
import json
from pathlib import Path

from m2_audit_zh_graph_cache_alignment import current_memories, signature


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
RETRIEVAL = ROOT / "results/analysis/m2_multi_entity_cohort_retrieval_20260927.json"
OUT = ROOT / "results/analysis/m2_conv42_nature_beauty_trace_20260927.json"
CASE = "locomo_conv-42_qa71"
SOURCE_IDS = ("D11:9", "D28:23")
DIAGNOSTIC_PREFIXES = ("3b72b542", "c2fc5c10", "e196eb2e", "4ea576c4", "56217cd3")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    source = next(x for x in json.loads(DATASET.read_text()) if x["sample_id"] == "conv-42")
    question = next(q for q in source["qa"] if q["question"] == "乔安娜和内特都欣赏什么的美？")
    assert set(question["evidence"]) == set(SOURCE_IDS)
    turns = []
    for key, rows in source["conversation"].items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        turns += [{"session_id": key, "dia_id": row["dia_id"], "speaker": row["speaker"],
                   "text": row["text"]} for row in rows if row["dia_id"] in SOURCE_IDS]
    assert {row["dia_id"] for row in turns} == set(SOURCE_IDS)
    memories = current_memories()
    user_memories = {mid: row for mid, row in memories.items()
                     if row["user_id"] == "zhfull:locomo:conv-42"}
    cohort = json.loads(RETRIEVAL.read_text())
    case = next(c for c in cohort["cases"] if c["case_id"] == CASE)
    diagnostic = []
    for prefix in DIAGNOSTIC_PREFIXES:
        matches = [(mid, row) for mid, row in user_memories.items() if mid.startswith(prefix)]
        assert len(matches) == 1, (prefix, len(matches))
        mid, row = matches[0]
        positions = {}
        for name, arm in case["arms"].items():
            positions[name] = {stage: next((r["rank"] for r in arm[stage]
                                            if r["id"] == mid), None)
                               for stage in ("vector", "reranked")}
        diagnostic.append({"id": mid, "session_id": row["session_id"],
                           "content": row["content"], "positions": positions,
                           "in_merged_top15": any(r["id"] == mid for r in case["merged_context"])})
    session_memory_counts = {sid: sum(row["session_id"] == sid for row in user_memories.values())
                             for sid in ("session_11", "session_28")}
    nature_memories = [{"id": mid, "session_id": row["session_id"], "content": row["content"]}
                       for mid, row in user_memories.items()
                       if "大自然" in row["content"] or "自然的美" in row["content"]]
    report = {
        "scope": "One source-annotated Chinese LoCoMo question; current Chroma and frozen three-query retrieval only; no model calls",
        "input_sha256": {"dataset": sha(DATASET), "retrieval": sha(RETRIEVAL)},
        "current_id_content_user_session_sha256": signature(memories),
        "question": question, "source_turns": sorted(turns, key=lambda x: x["dia_id"]),
        "current_user_memory_count": len(user_memories),
        "source_session_memory_counts": session_memory_counts,
        "nature_text_memory_count": len(nature_memories), "nature_text_memories": nature_memories,
        "diagnostic_memories": diagnostic,
        "frozen_final_ids": {"original": [r["id"] for r in case["arms"]["original"]["reranked"]],
                             "merged": [r["id"] for r in case["merged_context"]]},
        "limits": ["Current memory records do not contain dia_id provenance; semantic alignment to source turns is manual.",
                   "String matches and diagnostic memories do not exhaust all paraphrases in the corpus.",
                   "Frozen retrieval and current memory state are different snapshots; ID/content alignment is inspected here, not all embedding bytes.",
                   "No counterfactual extraction, rerank or answer is run."],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("source turns", len(turns), "current memories", len(user_memories),
          "nature text memories", len(nature_memories))


if __name__ == "__main__":
    main()
