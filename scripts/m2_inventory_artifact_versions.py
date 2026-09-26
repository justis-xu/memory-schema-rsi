"""Inventory non-JSONL artifacts and visible memory versions in Chinese runs."""

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/m2_artifact_version_inventory_20260926.json"
RUNS = sorted((ROOT / "results_zh").glob("zhfull*.jsonl"))
ARTIFACT_ROOTS = (
    "results", "results_zh", "experiments/schema_rsi/runs",
    "data/graph", "data/graph_zh", "data/extraction", "data/search",
    "data/vector_store", "data/vector_store_zh",
)


def signature(mapping):
    blob = json.dumps(sorted(mapping.items()), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def inspect_artifact(path):
    row = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
           "suffix": path.suffix or "[none]"}
    if path.suffix == ".json":
        try:
            value = json.loads(path.read_text())
            row["top_level_type"] = type(value).__name__
            row["top_level_items"] = len(value) if isinstance(value, (dict, list)) else None
            row["top_level_keys_sample"] = sorted(value)[:12] if isinstance(value, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            row["top_level_type"] = "unreadable_json"
    return row


def main():
    artifact_paths = {}
    for relative in ARTIFACT_ROOTS:
        for path in (ROOT / relative).rglob("*"):
            if path.is_file() and path != OUT and path.suffix != ".jsonl" and "__pycache__" not in path.parts:
                artifact_paths[str(path)] = path
    artifacts = [inspect_artifact(p) for p in sorted(artifact_paths.values())]
    by_suffix = Counter(a["suffix"] for a in artifacts)

    visible = {}
    graph = {}
    for path in RUNS:
        retrieved_map = {}
        graph_map = {}
        conflicts = 0
        for line in path.open():
            row = json.loads(line)
            for memory in row.get("retrieved_memories") or []:
                metadata = memory.get("metadata") or {}
                item = (memory["content"], metadata.get("session_id"),
                        metadata.get("session_date"), metadata.get("user_id"))
                previous = retrieved_map.setdefault(memory["id"], item)
                conflicts += previous != item
            for memory in row.get("graph_memories") or []:
                previous = graph_map.setdefault(memory["id"], memory["content"])
                conflicts += previous != memory["content"]
        name = str(path.relative_to(ROOT))
        visible[name] = {"ids": len(retrieved_map), "signature_sha256": signature(retrieved_map),
                         "intra_run_id_content_conflicts": conflicts}
        graph[name] = {"ids": len(graph_map), "signature_sha256": signature(graph_map)}
        visible[name]["_map"] = retrieved_map
        graph[name]["_map"] = graph_map
    all_visible_maps = [x.pop("_map") for x in visible.values()]
    all_graph_maps = [x.pop("_map") for x in graph.values()]
    same_visible = all(m == all_visible_maps[0] for m in all_visible_maps[1:])
    graph_union = {}
    graph_conflicts = 0
    for mapping in all_graph_maps:
        for mid, content in mapping.items():
            previous = graph_union.setdefault(mid, content)
            graph_conflicts += previous != content
    retrieved_union = all_visible_maps[0]
    graph_overlap = set(graph_union) & set(retrieved_union)
    cross_surface_conflicts = sum(graph_union[mid] != retrieved_union[mid][0] for mid in graph_overlap)
    payload = {
        "scope": "Non-JSONL files under specified run/result/cache roots; eight Chinese main runs' observed memory unions",
        "artifact_counts": {"files": len(artifacts), "by_suffix": dict(by_suffix)},
        "artifacts": artifacts,
        "visible_retrieved_memories": visible,
        "visible_retrieved_memory_maps_identical_across_eight_runs": same_visible,
        "graph_candidate_memories": graph,
        "cross_surface": {"retrieved_union_ids": len(retrieved_union), "graph_union_ids": len(graph_union),
                          "id_overlap": len(graph_overlap), "graph_only_ids": len(set(graph_union) - set(retrieved_union)),
                          "content_conflicts_on_overlap": cross_surface_conflicts,
                          "graph_content_conflicts_across_runs": graph_conflicts},
        "limits": ["Observed retrieval union is not a full memory-store snapshot",
                   "Artifacts can be reused or mutated after runs; filesystem mtime does not establish historical run provenance",
                   "Top-level JSON shape and file size do not validate scientific results"],
    }
    assert same_visible and len(retrieved_union) == 1780
    assert cross_surface_conflicts == graph_conflicts == 0
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"artifact_counts": payload["artifact_counts"], "visible_identical": same_visible,
                      "cross_surface": payload["cross_surface"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
