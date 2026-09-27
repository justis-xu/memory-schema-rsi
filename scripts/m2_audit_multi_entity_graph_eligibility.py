#!/usr/bin/env python3
"""Audit whether the frozen eight Chinese multi-person questions can test a cached graph bridge."""

import hashlib
import json
from pathlib import Path

from m2_audit_zh_graph_cache_alignment import current_memories, signature


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results/analysis/m2_multi_entity_cohort_retrieval_20260927.json"
STRUCTURED = ROOT / "data/graph_zh/structured_cache.json"
TOPIC = ROOT / "data/graph_zh/topic_cache.json"
EMBEDDING = ROOT / "data/graph_zh/memory_embeddings.json"
OUT = ROOT / "results/analysis/m2_multi_entity_graph_eligibility_20260927.json"

# Case evidence was selected after inspecting the archived answer and current memory text.
# The pair is a diagnostic for the *required fact*, not an oracle retrieval input.
FACT_PAIRS = {
    "locomo_conv-26_qa55": ("2b624cc0", "dbe74bc6"),
    "locomo_conv-30_qa18": ("5651babb", "5931c7af"),
    "locomo_conv-30_qa27": ("9fbb9b83", "cd69ce5e"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(prefix, ids):
    matches = [mid for mid in ids if mid.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError((prefix, len(matches)))
    return matches[0]


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    memories = current_memories()
    structured = json.loads(STRUCTURED.read_text())
    topic = json.loads(TOPIC.read_text())
    embedding = json.loads(EMBEDDING.read_text())
    cohort = json.loads(RETRIEVAL.read_text())
    cases = []
    for case in cohort["cases"]:
        user = case["user_id"]
        ids = {mid for mid, row in memories.items() if row["user_id"] == user}
        ranks = {row["id"]: row["rank"] for row in case["arms"]["original"]["reranked"]}
        pairs = []
        for prefix in FACT_PAIRS.get(case["case_id"], ()):
            mid = resolve(prefix, ids)
            ext = structured.get(mid, {})
            pairs.append({
                "id": mid,
                "memory_content": memories[mid]["content"],
                "original_top15_rank": ranks.get(mid),
                "structured_cached": mid in structured,
                "entity_names": [ent["name"] for ent in ext.get("entities", [])],
                "event_names": [event["name"] for event in ext.get("events", [])],
            })
        norm_sets = [{" ".join(name.lower().split()) for name in row["entity_names"]}
                     for row in pairs]
        cases.append({
            "case_id": case["case_id"], "question": case["question"], "user_id": user,
            "current_memory_count": len(ids),
            "structured_cached_count": len(ids & structured.keys()),
            "topic_cached_count": len(ids & topic.keys()),
            "embedding_cached_count": len(ids & embedding.keys()),
            "original_top15_structured_cached_count": len(set(ranks) & structured.keys()),
            "fact_pair": pairs,
            "fact_pair_exact_shared_entity_names": sorted(norm_sets[0] & norm_sets[1])
                if len(norm_sets) == 2 else None,
        })
    report = {
        "scope": "Eight previously question-text-selected Chinese two-person cases; current Chroma memory and cached graph extraction only; zero model or graph calls",
        "inputs_sha256": {"retrieval": sha(RETRIEVAL), "structured": sha(STRUCTURED),
                          "topic": sha(TOPIC), "embedding": sha(EMBEDDING)},
        "current_id_content_user_session_sha256": signature(memories),
        "selection_note": "Six required-fact memory exemplars in the three cached cases were manually selected after inspecting archived answers; they are diagnostics, not an unbiased sample or retrieval input.",
        "cases": cases,
        "limits": ["Cache coverage alone does not prove a traversable or source-faithful graph.",
                   "The two exemplar memory IDs per cached case do not exhaust possible graph paths.",
                   "Original rerank ranks are from a frozen run, while memory/cache state is read now.",
                   "No HugeGraph edge order, candidate cap, rerank-after-graph or answer effect is tested."],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in cases:
        print(row["case_id"], row["structured_cached_count"], "/",
              row["current_memory_count"], row["fact_pair_exact_shared_entity_names"])


if __name__ == "__main__":
    main()
