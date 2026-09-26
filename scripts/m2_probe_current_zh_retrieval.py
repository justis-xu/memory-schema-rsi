#!/usr/bin/env python3
"""Read-only current Chinese Mem0 retrieval probe for five source-backed cases."""

import hashlib
import json
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.llm.rerank import RerankClient
from schema_rsi.memory import Mem0Backend


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/analysis/m2_graph_fact_bridge_cases_20260926.json"
OUT = ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json"
VECTOR_DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
TARGETS = {
    "locomo_conv-47_qa96": ["6956a2b8-082c-4373-aa77-2c925378ddf3"],
    "locomo_conv-48_qa39": ["6bec2654-3991-4a1d-ad50-1c1f8ecf762e"],
    "locomo_conv-43_qa25": ["1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc", "219b4c97-f2af-4734-a4f4-1d93ebe19f99"],
    "locomo_conv-41_qa86": ["dccbdd97-4ac6-4028-8068-840070dcabff"],
    "locomo_conv-30_qa80": ["d4810a3f-ef1f-4778-bd1a-49b2200f7cc8"],
}


def db_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def serialize(record, rank, score=None):
    return {
        "rank": rank, "id": record.id, "content": record.content,
        "session_id": record.metadata.get("session_id"),
        "session_date": record.metadata.get("session_date"),
        "user_id": record.metadata.get("user_id"),
        "rerank_score": score,
    }


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    packet = {case["case_id"]: case for case in json.loads(PACKET.read_text())["cases"]}
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    backend = Mem0Backend(settings)
    reranker = RerankClient(settings.rerank.base_url, settings.rerank.api_key, settings.rerank.model)
    result = {
        "scope": "Five fixed Chinese questions, current Mem0 zhfull users; one vector search and one rerank operation per question",
        "vector_db_sha256_before": db_hash(VECTOR_DB),
        "embedding_model": settings.embedding.model,
        "rerank_model": settings.rerank.model,
        "top_k_vector": 45, "top_k_final": 15,
        "cases": [],
        "limits": [
            "Current vector store is later than old graph runs and is not an immutable snapshot after this probe.",
            "Five retrospectively selected questions do not estimate current overall recall or answer accuracy.",
            "This bypasses any future atomic pool; current SQLite has no schema_rsi_atomic collection.",
        ],
    }
    for case_id, target_ids in TARGETS.items():
        row = packet[case_id]
        conv_id = case_id.removeprefix("locomo_").split("_qa")[0]
        user_id = f"zhfull:locomo:{conv_id}"
        vector = backend.search_memory(row["question"], user_id=user_id, top_k=45)
        assert all(m.metadata.get("user_id") == user_id for m in vector)
        scored = reranker.rerank(row["question"], [m.content for m in vector], top_n=15)
        final = [vector[item["index"]] for item in scored if 0 <= item["index"] < len(vector)][:15]
        case = {
            "case_id": case_id, "question": row["question"], "gold": row["gold"],
            "user_id": user_id, "target_memory_ids": target_ids,
            "vector": [serialize(m, i) for i, m in enumerate(vector, 1)],
            "reranked": [serialize(m, i, scored[i-1]["score"]) for i, m in enumerate(final, 1)],
            "target_vector_ranks": {mid: next((i for i, m in enumerate(vector, 1) if m.id == mid), None) for mid in target_ids},
            "target_rerank_ranks": {mid: next((i for i, m in enumerate(final, 1) if m.id == mid), None) for mid in target_ids},
            "rerank_endpoint_style": reranker.last_endpoint,
        }
        result["cases"].append(case)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(case_id, case["target_vector_ranks"], case["target_rerank_ranks"], flush=True)
    result["vector_db_sha256_after"] = db_hash(VECTOR_DB)
    result["vector_db_unchanged"] = result["vector_db_sha256_before"] == result["vector_db_sha256_after"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}; vector_db_unchanged={result['vector_db_unchanged']}")


if __name__ == "__main__":
    main()
