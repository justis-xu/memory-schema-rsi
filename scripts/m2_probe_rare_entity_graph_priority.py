#!/usr/bin/env python3
"""Isolated current Chinese graph candidate ordering stress test, no services."""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m2_audit_zh_graph_cache_alignment import current_memories, signature  # noqa: E402
from m2_replay_signed_ball_graph import LocalGraph  # noqa: E402
from schema_rsi.evaluation.retriever import GraphRetriever  # noqa: E402
from schema_rsi.graph.builder import GraphBuilder  # noqa: E402
from schema_rsi.memory.base import MemoryRecord  # noqa: E402


USER = "zhfull:locomo:conv-43"
JOHN = "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"
CASE_IDS = ("locomo_conv-43_qa25", "locomo_conv-43_qa2", "locomo_conv-43_qa32")
CAP = 30
RARE_MAX_DEGREE = 3
OUT = ROOT / "results/analysis/m2_rare_entity_graph_priority_20260927.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_inputs():
    old = json.loads((ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json").read_text())
    recent = json.loads((ROOT / "results/analysis/m2_current_zh_stratified_retrieval_20260927.json").read_text())
    lookup = {row["case_id"]: row for result in (old, recent) for row in result["cases"]}
    return {cid: {"question": lookup[cid]["question"],
                  "seed_ids": [m["id"] for m in lookup[cid]["reranked"][:5]],
                  "exclude_ids": {m["id"] for m in lookup[cid]["reranked"]}}
            for cid in CASE_IDS}


def entity_degrees(store):
    counts = Counter()
    for edge_label, out, target in store.edges:
        if edge_label == "MENTIONS" and out[0] == "Memory" and target[0] == "Entity":
            counts[target] += 1
    by_tag = {}
    for node, count in counts.items():
        name = store.vertices[node].get("name") or node[1]
        tag = f"entity:{name}"
        by_tag[tag] = max(by_tag.get(tag, 0), count)
    return by_tag


def priority(row, degrees):
    return any(degrees.get(via, 10**9) <= RARE_MAX_DEGREE
               for via in row["via"] if via.startswith("entity:"))


def one_run(order, records, structured, topics, cases, *, detailed=False):
    store = LocalGraph()
    builder = GraphBuilder(store)
    recs = [records[mid] for mid in order]
    builder.build_s1(recs, structured, reset=True)
    builder.build_concepts(recs, topics)
    degrees = entity_degrees(store)
    output = {}
    for cid, spec in cases.items():
        seeds = [records[mid] for mid in spec["seed_ids"]]
        found = GraphRetriever(store).retrieve_fused(
            seeds, exclude_ids=spec["exclude_ids"], per_node_limit=15,
            max_candidates=500)
        assert len(found) < 500
        reordered = sorted(enumerate(found), key=lambda pair: (not priority(pair[1], degrees), pair[0]))
        ranked = [row for _, row in reordered]
        base_ids = [row["id"] for row in found[:CAP]]
        proposed_ids = [row["id"] for row in ranked[:CAP]]
        result = {"pool_size": len(found), "rare_candidates": sum(priority(row, degrees) for row in found),
                       "baseline_ids": base_ids, "rare_first_ids": proposed_ids,
                       "entered_ids": sorted(set(proposed_ids) - set(base_ids)),
                       "exited_ids": sorted(set(base_ids) - set(proposed_ids)),
                       "john_baseline_rank": next((i for i, row in enumerate(found, 1) if row["id"] == JOHN), None),
                       "john_rare_first_rank": next((i for i, row in enumerate(ranked, 1) if row["id"] == JOHN), None)}
        if detailed:
            result.update({"baseline_top30": [{"id": row["id"], "content": row["content"],
                                           "via": row["via"], "support": row["support"],
                                           "rare_entity": priority(row, degrees)} for row in found[:CAP]],
                       "rare_first_top30": [{"id": row["id"], "content": row["content"],
                                            "via": row["via"], "support": row["support"],
                                            "rare_entity": priority(row, degrees)} for row in ranked[:CAP]]})
        output[cid] = result
    return output


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    db_path = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
    db_before = sha(db_path)
    all_memory = current_memories()
    subset = {mid: m for mid, m in all_memory.items() if m["user_id"] == USER}
    assert len(subset) == 211
    records = {mid: MemoryRecord(id=mid, content=m["content"],
                                 metadata={"user_id": USER, "session_id": m["session_id"],
                                           "session_date": m.get("session_date")})
               for mid, m in subset.items()}
    structured_path = ROOT / "data/graph_zh/structured_cache.json"
    topics_path = ROOT / "data/graph_zh/topic_cache.json"
    structured = json.loads(structured_path.read_text())
    topics = json.loads(topics_path.read_text())
    cases = case_inputs()
    ids = sorted(records)
    orders = {"sorted": ids, "reversed": list(reversed(ids))}
    for seed in range(50):
        shuffled = ids[:]
        random.Random(seed).shuffle(shuffled)
        orders[f"shuffle_{seed}"] = shuffled
    runs = {name: one_run(order, records, structured, topics, cases,
                          detailed=name in ("sorted", "reversed"))
            for name, order in orders.items()}
    result = {"scope": "Current conv-43 cached S2 graph, three frozen Chinese questions, same seeds/per-node=15/candidate cap=30; no HugeGraph, reranker or model calls",
              "policy": "Stable partition: candidates with an Entity via node of full degree <=3 first, preserving production support/order inside each partition; no question gold used",
              "rare_max_degree": RARE_MAX_DEGREE, "candidate_cap": CAP,
              "case_questions": {cid: spec["question"] for cid, spec in cases.items()},
              "seed_ids": {cid: spec["seed_ids"] for cid, spec in cases.items()},
              "current_chroma_logic_sha256": signature(all_memory),
              "chroma_file_sha256_before": db_before, "chroma_file_sha256_after": sha(db_path),
              "structured_cache_sha256": sha(structured_path), "topic_cache_sha256": sha(topics_path),
              "runs": runs,
              "summary": {cid: {"orders": len(runs),
                                 "baseline_john_in_top30": sum(v[cid]["john_baseline_rank"] is not None and v[cid]["john_baseline_rank"] <= CAP for v in runs.values()),
                                 "rare_first_john_in_top30": sum(v[cid]["john_rare_first_rank"] is not None and v[cid]["john_rare_first_rank"] <= CAP for v in runs.values()),
                                 "orders_with_candidate_set_change": sum(bool(v[cid]["entered_ids"]) for v in runs.values()),
                                 "total_entered_candidates": sum(len(v[cid]["entered_ids"]) for v in runs.values())}
                          for cid in cases},
              "limits": ["Rare entity degree is computed on current cached in-process graph; caches lack source text hashes and may omit current memories",
                         "Order permutations are stress tests, not independent questions or online success frequencies",
                         "Candidate cap exposure does not establish final fused rerank entry, answer benefit or harm"]}
    assert result["chroma_file_sha256_before"] == result["chroma_file_sha256_after"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["summary"])


if __name__ == "__main__":
    main()
