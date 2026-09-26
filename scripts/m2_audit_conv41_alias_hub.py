"""Read-only structural counterfactual for conv-41 User/John/约翰 graph aliases."""

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
CACHE = ROOT / "data/graph_zh/structured_cache.json"
PACKET = ROOT / "results/analysis/m2_cross_session_carryover_packet_20260926.json"
OUTPUT = ROOT / "results/analysis/m2_conv41_alias_hub_20260926.json"

# Manual source-turn review of the five graph-cache memories named User.
SOURCE_IDS = {
    "ee7c0f33-94df-4d44-b23b-929f607032e7": ["D23:1"],
    "c10122e0-ec23-4065-81ad-4072971a5686": ["D23:3", "D23:4"],
    "d0b66813-5e5e-4dca-914e-c7510aeb6281": ["D23:5", "D23:6"],
    "fe0f098e-5c50-46d7-8f9a-6aa7194740bd": ["D22:5", "D23:7"],
    "3a4760bc-5e23-43a0-8e00-499b54e0232c": ["D22:5", "D22:7", "D22:9"],
}


def main():
    memories = {}
    for line in ARCHIVE.open():
        row = json.loads(line)
        if not row["case_id"].startswith("locomo_conv-41_"):
            continue
        for memory in row["retrieved_memories"]:
            old = memories.get(memory["id"])
            assert old is None or old["content"] == memory["content"]
            memories[memory["id"]] = memory
    cache = json.loads(CACHE.read_text())
    visible_cache = set(memories) & set(cache)
    alias_to_memories = defaultdict(set)
    for mid in visible_cache:
        for entity in cache[mid].get("entities", []):
            alias_to_memories[entity["name"]].add(mid)
    user_ids = alias_to_memories["User"]
    assert user_ids == set(SOURCE_IDS)

    packet = json.loads(PACKET.read_text())
    photo_pair = next(p for p in packet["pairs"] if p["pair_id"] == "conv-41_051a88fcd77c")
    turns = {t["dia_id"]: (sid, t) for sid, sess in photo_pair["source_sessions"].items() for t in sess["turns"]}
    rows = []
    for mid in sorted(user_ids):
        memory = memories[mid]
        refs = []
        for did in SOURCE_IDS[mid]:
            sid, turn = turns[did]
            refs.append({"session_id": sid, "dia_id": did, "speaker": turn["speaker"], "text": turn["text"]})
        rows.append({"memory_id": mid, "stored_session_id": memory.get("metadata", {}).get("session_id"),
                     "content": memory["content"], "source_turns": refs,
                     "has_john_source_turn": any(r["speaker"] == "约翰" for r in refs)})

    john = alias_to_memories["John"]
    john_zh = alias_to_memories["约翰"]
    assert not (user_ids & john) and not (user_ids & john_zh)
    result = {
        "scope": "conv-41 memories visible in the old Chinese no-graph archive and matching current structured extraction cache",
        "counts": {"visible_memory_ids": len(memories), "cache_matched_ids": len(visible_cache),
                   "user_entity_memories": len(user_ids), "john_entity_memories": len(john),
                   "john_zh_entity_memories": len(john_zh),
                   "user_to_john_additional_structural_neighbors_per_user_seed": len(john - user_ids),
                   "user_to_john_zh_additional_structural_neighbors_per_user_seed": len(john_zh - user_ids),
                   "all_three_alias_union_memories": len(user_ids | john | john_zh)},
        "user_memories": rows,
        "limits": ["Only graph-cache MENTIONS structure is simulated; no HugeGraph query, per-node limit, rerank, final context or answer replay",
                   "Archive retrieval union is not the complete memory store; cache and archive are close but not identical versions",
                   "Source-turn IDs are manually selected; an alias merge does not correct stale session metadata or unsupported details"],
    }
    assert all(r["has_john_source_turn"] for r in rows)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
