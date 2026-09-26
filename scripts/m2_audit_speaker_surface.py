#!/usr/bin/env python3
"""只读中文旧归档与结构缓存，盘点泛称人物是否大量进入记忆/图输入。"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
RAW = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
STRUCTURED = ROOT / "data/graph_zh/structured_cache.json"
OUTPUT = ROOT / "results/analysis/m2_speaker_surface_20260926.json"
GENERIC_PREFIX = re.compile(r"^(?:用户|助手|对话中的|User\b|The user\b|Assistant\b|The assistant\b)", re.I)
GENERIC_ENTITY = re.compile(r"^(?:用户|助手|助理|User|The user|Assistant|The assistant|我|他|她)$", re.I)
GENERIC_REFERENCE = re.compile(r"\b(?:User|Assistant)\b|用户|助手|助理", re.I)


def main() -> None:
    dataset = {entry["sample_id"]: entry for entry in json.loads(DATASET.read_text())}
    by_conv: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in RAW.open():
        if not line.strip():
            continue
        row = json.loads(line)
        conv_id = row["case_id"].split("_")[1]
        for memory in row["retrieved_memories"]:
            prior = by_conv[conv_id].get(memory["id"])
            if prior and prior["content"] != memory["content"]:
                raise ValueError(f"ID content drift: {memory['id']}")
            by_conv[conv_id][memory["id"]] = memory
    memory_rows = []
    memory_counts = Counter()
    for conv_id, memories in sorted(by_conv.items()):
        conv = dataset[conv_id]["conversation"]
        memory_counts["visible_unique_ids"] += len(memories)
        for memory in memories.values():
            content = memory["content"].strip()
            has_prefix = bool(GENERIC_PREFIX.search(content))
            if not GENERIC_REFERENCE.search(content):
                continue
            memory_counts["generic_anywhere_ids"] += 1
            memory_counts["generic_prefix_ids"] += has_prefix
            memory_rows.append({
                "conversation": conv_id,
                "speaker_a": conv.get("speaker_a"),
                "speaker_b": conv.get("speaker_b"),
                "memory_id": memory["id"],
                "session_id": (memory.get("metadata") or {}).get("session_id"),
                "generic_at_start": has_prefix,
                "content": memory["content"],
            })
    graph_counts = Counter()
    graph_rows = []
    focus_names = Counter()
    if STRUCTURED.exists():
        cache = json.loads(STRUCTURED.read_text())
        graph_counts["cache_memory_ids"] = len(cache)
        archive_id_to_conv = {
            memory_id: conv_id for conv_id, memories in by_conv.items()
            for memory_id in memories
        }
        graph_counts["cache_ids_in_archive_union"] = len(set(cache) & set(archive_id_to_conv))
        for memory_id, facts in cache.items():
            for entity in facts.get("entities", []):
                graph_counts["entity_mentions"] += 1
                if archive_id_to_conv.get(memory_id) == "conv-41" and entity.get("name") in {"John", "User", "约翰"}:
                    focus_names[entity["name"]] += 1
                if GENERIC_ENTITY.fullmatch(str(entity.get("name", "")).strip()):
                    graph_counts["generic_entity_mentions"] += 1
                    graph_rows.append({"memory_id": memory_id, "kind": "entity", "value": entity})
            for relation in facts.get("relationships", []):
                graph_counts["relationship_mentions"] += 1
                if any(GENERIC_REFERENCE.search(str(value)) for value in relation.values()):
                    graph_counts["relationships_with_generic_reference"] += 1
                    graph_rows.append({"memory_id": memory_id, "kind": "relationship", "value": relation})
    report = {
        "scope": "2026-09-25 中文无图旧归档检索并集的唯一向量记忆 ID；另看当前结构提取缓存（版本可能不同）",
        "method": "词面查记忆正文任何位置的 User/Assistant/用户/助手/助理，并另记开头泛称；结构缓存查精确泛称实体和含泛称的关系。不进行语义归属判定。",
        "memory_counts": dict(memory_counts),
        "memories_with_generic_reference": memory_rows,
        "graph_cache_counts": dict(graph_counts),
        "conv41_entity_alias_mentions": dict(focus_names),
        "graph_cache_generic_items": graph_rows,
        "limits": [
            "旧归档检索并集不是完整历史 Mem0 快照；当前结构缓存不一定与旧归档版本完全一致。",
            "泛称少不证明人物归属正确；命名记忆也可能把事实归错人。",
            "User 在单对话中可能是 speaker_a 的可解释别名，不自动计为错误。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"memory_counts": report["memory_counts"],
                      "graph_cache_counts": report["graph_cache_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
