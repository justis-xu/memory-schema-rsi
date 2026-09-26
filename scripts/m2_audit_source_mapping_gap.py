#!/usr/bin/env python3
"""核对中文归档记忆是否带精确原始 turn 来源；零模型调用。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "baseline": ROOT / "results_zh/zhfull_locomo_20260925_164645.jsonl",
    "fused_graph": ROOT / "results_zh/zhfull_fused_locomo_20260925_172126.jsonl",
}
OUTPUT = ROOT / "results/analysis/m2_source_mapping_gap_20260926.json"
TURN_KEYS = {"dia_id", "dia_ids", "turn_id", "turn_ids", "source_turn_id", "source_turn_ids"}


def inspect(path: Path) -> dict:
    counts = Counter()
    metadata_keys = Counter()
    vector_ids = {}
    graph_ids = {}
    selected_graph_ids = set()
    selected_vector_ids = set()
    rows_missing_context_id = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["question_rows"] += 1
            vector = {m["id"]: m for m in row["retrieved_memories"]}
            graph = {m["id"]: m for m in row.get("graph_memories", [])}
            for memory in row["retrieved_memories"]:
                counts["vector_appearances"] += 1
                metadata = memory.get("metadata") or {}
                metadata_keys.update(metadata.keys())
                counts["vector_appearances_with_session_id"] += bool(metadata.get("session_id"))
                counts["vector_appearances_with_session_date"] += bool(metadata.get("session_date"))
                counts["vector_appearances_with_turn_id"] += any(metadata.get(k) for k in TURN_KEYS)
                old = vector_ids.setdefault(memory["id"], memory)
                if old["content"] != memory["content"]:
                    raise AssertionError(f"vector ID changed content: {memory['id']}")
            for memory in row.get("graph_memories", []):
                counts["graph_candidate_appearances"] += 1
                graph_ids.setdefault(memory["id"], memory)
            for memory_id in row.get("metadata", {}).get("answer_context_ids") or []:
                if memory_id in graph and memory_id not in vector:
                    counts["selected_graph_appearances"] += 1
                    selected_graph_ids.add(memory_id)
                elif memory_id in vector:
                    counts["selected_vector_appearances"] += 1
                    selected_vector_ids.add(memory_id)
                else:
                    rows_missing_context_id.append(row["case_id"])
    if rows_missing_context_id:
        raise AssertionError(f"missing selected IDs in {path}: {rows_missing_context_id[:5]}")

    unique_vector = {
        "distinct_ids": len(vector_ids),
        "with_session_id": sum(bool((m.get("metadata") or {}).get("session_id")) for m in vector_ids.values()),
        "with_session_date": sum(bool((m.get("metadata") or {}).get("session_date")) for m in vector_ids.values()),
        "with_turn_id": sum(any((m.get("metadata") or {}).get(k) for k in TURN_KEYS) for m in vector_ids.values()),
    }
    graph_metadata = Counter()
    for memory in graph_ids.values():
        graph_metadata.update((memory.get("metadata") or {}).keys())
    return {
        "path": str(path.relative_to(ROOT)),
        "counts": dict(counts),
        "vector_metadata_keys": dict(metadata_keys),
        "unique_vector": unique_vector,
        "unique_graph_candidate_ids": len(graph_ids),
        "unique_selected_graph_ids": len(selected_graph_ids),
        "unique_selected_vector_ids": len(selected_vector_ids),
        "graph_metadata_keys": dict(graph_metadata),
    }


def main() -> None:
    report = {
        "scope": "2026-09-25 中文无图与融合图全量原始归档中可见的记忆元数据；零模型调用",
        "turn_id_keys_checked": sorted(TURN_KEYS),
        "arms": {name: inspect(path) for name, path in INPUTS.items()},
        "limits": [
            "只能断言归档检索记录和最终上下文缺精确 turn ID；不证明底层其他系统绝无来源信息。",
            "一条记忆可能由同会话多个 turn 合成，单一 dia_id 字段未必足够，应保存来源 turn 列表及支持关系。",
            "graph 候选与 vector 同 ID 可从 vector 补部分会话元数据，但无法凭 session_id 还原精确来源 turn。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {"counts": arm["counts"], "unique_vector": arm["unique_vector"],
                             "graph_ids": arm["unique_graph_candidate_ids"]}
                      for name, arm in report["arms"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
