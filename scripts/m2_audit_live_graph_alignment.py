#!/usr/bin/env python3
"""Read-only spot audit: do current Chinese vector IDs resolve in live HugeGraph?"""

import json
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.graph.hugegraph_store import HugeGraphStore


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json"
OUT = ROOT / "results/analysis/m2_live_graph_alignment_20260927.json"
JOHN_SIGNED_BALL_ID = "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    retrieval = json.loads(RETRIEVAL.read_text())
    selected = []
    for case in retrieval["cases"]:
        for row in case["reranked"][:3]:
            selected.append({"case_id": case["case_id"], "id": row["id"],
                             "vector_content": row["content"], "vector_user_id": row["user_id"]})
    assert len(selected) == 15
    # One additional current ID known to exist in the vector store but absent
    # from the signed-ball original query's vector top 45.
    selected.append({"case_id": "locomo_conv-43_qa25", "id": JOHN_SIGNED_BALL_ID,
                     "vector_content": None, "vector_user_id": "zhfull:locomo:conv-43"})
    store = HugeGraphStore(get_settings(str(ROOT / "config/locomo_zh.yaml")))
    info = store.server_info()
    one_memory = store._request("GET", f"/graphs/{store.graph}/graph/vertices",
                                params={"label": "Memory", "limit": 1})
    sample_vertices = (one_memory or {}).get("vertices", [])
    sample_props = sample_vertices[0].get("properties", {}) if sample_vertices else {}
    result = {
        "scope": "Five previously fixed Chinese cases, first three current reranked vector IDs each, plus one known signed-ball ID; live graph read-only",
        "source_retrieval": str(RETRIEVAL.relative_to(ROOT)),
        "graph_response": {"graphs": info.get("graphs"), "versions": info.get("versions")},
        "loaded_graph_key_props": dict(store._key_props),
        "one_live_memory_sample": {key: sample_props.get(key) for key in ("memory_id", "user_id", "content")},
        "cases": [],
        "limits": ["Sixteen deterministic IDs are a spot audit, not a full graph inventory", "A running server and matching graph name do not prove matching memory version", "No graph rebuild or model calls"],
    }
    for row in selected:
        vertex = store.get_vertex("Memory", row["id"])
        props = (vertex or {}).get("properties") or {}
        result["cases"].append({
            **row,
            "live_vertex_found": vertex is not None,
            "live_content_matches_vector": props.get("content") == row["vector_content"] if vertex and row["vector_content"] is not None else None,
            "live_user_matches_vector": props.get("user_id") == row["vector_user_id"] if vertex else None,
            "live_content": props.get("content") if vertex else None,
            "live_user_id": props.get("user_id") if vertex else None,
        })
    result["found_count"] = sum(row["live_vertex_found"] for row in result["cases"])
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"found {result['found_count']}/{len(selected)} selected current Memory IDs")


if __name__ == "__main__":
    main()
