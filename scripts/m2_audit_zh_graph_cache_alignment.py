#!/usr/bin/env python3
"""Read-only full-ID alignment of current Chinese memory and graph caches."""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
CACHES = {name: ROOT / f"data/graph_zh/{name}_cache.json"
          for name in ("structured", "topic")}
CACHES["memory_embeddings"] = ROOT / "data/graph_zh/memory_embeddings.json"
OUT = ROOT / "results/analysis/m2_zh_graph_cache_alignment_20260927.json"
TARGETS = {"Tim": "219b4c97-f2af-4734-a4f4-1d93ebe19f99",
           "John": "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"}


def current_memories():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("""
        select e.embedding_id,
          max(case when m.key='data' then m.string_value end),
          max(case when m.key='user_id' then m.string_value end),
          max(case when m.key='session_id' then m.string_value end)
        from embeddings e left join embedding_metadata m on m.id=e.id
        group by e.id order by e.embedding_id
    """)
    return {mid: {"content": content, "user_id": user, "session_id": session}
            for mid, content, user, session in rows}


def signature(memories):
    h = hashlib.sha256()
    for mid in sorted(memories):
        row = memories[mid]
        h.update(json.dumps((mid, row["content"], row["user_id"], row["session_id"]),
                            ensure_ascii=False, separators=(",", ":")).encode())
    return h.hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    memories = current_memories()
    ids = set(memories)
    zhfull_ids = {mid for mid, row in memories.items()
                  if (row["user_id"] or "").startswith("zhfull:locomo:")}
    by_user = Counter(row["user_id"] for row in memories.values())
    cache_data = {name: json.loads(path.read_text()) for name, path in CACHES.items()}
    result = {
        "scope": "Current Chinese Chroma IDs vs three graph caches; no model or graph calls",
        "current_memory_count": len(ids),
        "current_zhfull_memory_count": len(zhfull_ids),
        "current_id_content_user_session_sha256": signature(memories),
        "current_users": dict(sorted(by_user.items())),
        "caches": {}, "selected_pair": {},
        "limits": ["Caches keyed only by memory ID; cache membership does not prove extracted claims are source faithful", "Current Chroma content may have changed while an ID stayed the same; caches have no source content hash", "No isolated graph was constructed"],
    }
    for name, path in CACHES.items():
        keys = set(cache_data[name])
        result["caches"][name] = {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cache_key_count": len(keys),
            "current_id_covered_count": len(keys & ids),
            "current_id_missing_count": len(ids - keys),
            "current_zhfull_id_covered_count": len(keys & zhfull_ids),
            "current_zhfull_id_missing_count": len(zhfull_ids - keys),
            "stale_key_count": len(keys - ids),
            "missing_by_user": dict(sorted(Counter(memories[mid]["user_id"] for mid in ids - keys).items())),
        }
    for person, mid in TARGETS.items():
        result["selected_pair"][person] = {
            "id": mid, "current_memory": memories.get(mid),
            "structured": cache_data["structured"].get(mid),
            "topics": cache_data["topic"].get(mid),
            "embedding_cached": mid in cache_data["memory_embeddings"],
        }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, row in result["caches"].items():
        print(name, row["current_id_covered_count"], '/', len(ids),
              'stale', row["stale_key_count"])


if __name__ == "__main__":
    main()
