#!/usr/bin/env python3
"""Compare two non-oracle person-split queries with the current graph-cache bridge."""

import hashlib
import json
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.llm.rerank import RerankClient
from schema_rsi.memory import Mem0Backend


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json"
OUT = ROOT / "results/analysis/m2_signed_ball_paths_20260926.json"
VECTOR_DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
TIM_ID = "219b4c97-f2af-4734-a4f4-1d93ebe19f99"
JOHN_ID = "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"
QUERIES = ["约翰有哪些运动收藏品？", "蒂姆有哪些运动收藏品？"]


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(m, rank, score=None):
    return {"rank": rank, "id": m.id, "content": m.content,
            "session_id": m.metadata.get("session_id"), "rerank_score": score}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    baseline = json.loads(BASELINE.read_text())
    db_before = hash_file(VECTOR_DB)
    assert db_before == baseline["vector_db_sha256_after"]
    cache = json.loads((ROOT / "data/graph_zh/structured_cache.json").read_text())
    related = sorted(mid for mid, item in cache.items()
                     if any(e.get("name") == "signed basketball" for e in item.get("entities", [])))
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    backend = Mem0Backend(settings)
    reranker = RerankClient(settings.rerank.base_url, settings.rerank.api_key, settings.rerank.model)
    result = {
        "scope": "Two person-split queries derived only from the original Chinese question; current zhfull conv-43 store",
        "original_question": "蒂姆和约翰拥有什么相似的运动收藏品？",
        "user_id": "zhfull:locomo:conv-43",
        "queries": [],
        "current_cache_signed_basketball_entity_memory_ids": related,
        "target_ids": {"Tim": TIM_ID, "John": JOHN_ID},
        "vector_db_sha256_before": db_before,
        "limits": ["Graph cache entity co-occurrence is a potential path, not proof of a live graph traversal", "Two hand-written non-oracle subqueries are a local query-reformulation probe, not a trained general method", "No answer model or judge calls"],
    }
    for query in QUERIES:
        vector = backend.search_memory(query, user_id=result["user_id"], top_k=45)
        assert all(m.metadata.get("user_id") == result["user_id"] for m in vector)
        scored = reranker.rerank(query, [m.content for m in vector], top_n=15)
        final = [vector[s["index"]] for s in scored if 0 <= s["index"] < len(vector)][:15]
        row = {
            "query": query,
            "vector": [record(m, i) for i, m in enumerate(vector, 1)],
            "reranked": [record(m, i, scored[i-1]["score"]) for i, m in enumerate(final, 1)],
            "target_vector_ranks": {mid: next((i for i, m in enumerate(vector, 1) if m.id == mid), None) for mid in (TIM_ID, JOHN_ID)},
            "target_rerank_ranks": {mid: next((i for i, m in enumerate(final, 1) if m.id == mid), None) for mid in (TIM_ID, JOHN_ID)},
            "rerank_endpoint_style": reranker.last_endpoint,
        }
        result["queries"].append(row)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(query, row["target_vector_ranks"], row["target_rerank_ranks"], flush=True)
    result["vector_db_sha256_after"] = hash_file(VECTOR_DB)
    result["vector_db_unchanged"] = result["vector_db_sha256_before"] == result["vector_db_sha256_after"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}; vector_db_unchanged={result['vector_db_unchanged']}")


if __name__ == "__main__":
    main()
