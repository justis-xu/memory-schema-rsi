#!/usr/bin/env python3
"""核对中文归档记忆是否带精确原始 turn 来源；零模型调用。"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "baseline": ROOT / "results_zh/zhfull_locomo_20260925_164645.jsonl",
    "fused_graph": ROOT / "results_zh/zhfull_fused_locomo_20260925_172126.jsonl",
}
OUTPUT = ROOT / "results/analysis/m2_source_mapping_gap_20260926.json"
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
TURN_KEYS = {"dia_id", "dia_ids", "turn_id", "turn_ids", "source_turn_id", "source_turn_ids"}


def source_sessions() -> dict[tuple[str, str], dict]:
    result = {}
    for entry in json.loads(DATASET.read_text()):
        for session_id, turns in entry["conversation"].items():
            if re.fullmatch(r"session_\d+", session_id) and isinstance(turns, list):
                result[(entry["sample_id"], session_id)] = {
                    "turn_count": len(turns),
                    "query_turn_count": sum(bool(turn.get("query")) for turn in turns),
                    "caption_turn_count": sum(bool(turn.get("blip_caption")) for turn in turns),
                }
    return result


def inspect(path: Path, sessions: dict[tuple[str, str], dict]) -> dict:
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
    linked_sessions = []
    for memory in vector_ids.values():
        metadata = memory["metadata"]
        conv_id = metadata["user_id"].rsplit(":", 1)[-1]
        linked_sessions.append(sessions[(conv_id, metadata["session_id"])])
    lengths = sorted(session["turn_count"] for session in linked_sessions)
    ambiguity = {
        "distinct_retrieved_memory_ids": len(lengths),
        "source_session_min_turns": lengths[0],
        "source_session_median_turns": statistics.median(lengths),
        "source_session_p90_turns_nearest_rank": lengths[int(0.9 * (len(lengths) - 1))],
        "source_session_max_turns": lengths[-1],
        "memories_from_sessions_with_query": sum(s["query_turn_count"] >= 1 for s in linked_sessions),
        "memories_from_sessions_with_at_least_two_queries": sum(s["query_turn_count"] >= 2 for s in linked_sessions),
        "memories_from_sessions_with_caption": sum(s["caption_turn_count"] >= 1 for s in linked_sessions),
    }
    graph_metadata = Counter()
    for memory in graph_ids.values():
        graph_metadata.update((memory.get("metadata") or {}).keys())
    return {
        "path": str(path.relative_to(ROOT)),
        "counts": dict(counts),
        "vector_metadata_keys": dict(metadata_keys),
        "unique_vector": unique_vector,
        "source_session_ambiguity": ambiguity,
        "unique_graph_candidate_ids": len(graph_ids),
        "unique_selected_graph_ids": len(selected_graph_ids),
        "unique_selected_vector_ids": len(selected_vector_ids),
        "graph_metadata_keys": dict(graph_metadata),
    }


def main() -> None:
    sessions = source_sessions()
    report = {
        "scope": "2026-09-25 中文无图与融合图全量原始归档中可见的记忆元数据；零模型调用",
        "source_dataset": str(DATASET),
        "turn_id_keys_checked": sorted(TURN_KEYS),
        "arms": {name: inspect(path, sessions) for name, path in INPUTS.items()},
        "limits": [
            "只能断言归档检索记录和最终上下文缺精确 turn ID；不证明底层其他系统绝无来源信息。",
            "一条记忆可能由同会话多个 turn 合成，单一 dia_id 字段未必足够，应保存来源 turn 列表及支持关系。",
            "graph 候选与 vector 同 ID 可从 vector 补部分会话元数据，但无法凭 session_id 还原精确来源 turn。",
            "某记忆所属会话有 query/caption，不表示这条记忆必然提取自该字段；计数只量化来源定位的歧义范围。",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {"counts": arm["counts"], "unique_vector": arm["unique_vector"],
                             "source_session_ambiguity": arm["source_session_ambiguity"],
                             "graph_ids": arm["unique_graph_candidate_ids"]}
                      for name, arm in report["arms"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
