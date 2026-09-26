"""Trace six graph-candidate IDs absent from the archived vector retrieval union."""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results_zh"
VECTOR_DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
HISTORY_DB = ROOT / "data/vector_store_zh/mem0_history_zh.db"
SOURCE = Path("/Users/xu/git/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_graph_only_memory_ids_20260926.json"
PRIOR_AUDIT = ROOT / "results/analysis/graph_context_audit_20260926.json"


def readonly(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def main():
    vector_seen = {}
    graph_seen = {}
    occurrences = defaultdict(list)
    for path in sorted(RUN_DIR.glob("zhfull*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            context_ids = set(row.get("metadata", {}).get("answer_context_ids") or [])
            for mem in row.get("retrieved_memories") or []:
                vector_seen[mem["id"]] = mem["content"]
            for mem in row.get("graph_memories") or []:
                previous = graph_seen.setdefault(mem["id"], mem["content"])
                assert previous == mem["content"]
                occurrences[mem["id"]].append({
                    "run": path.name, "case_id": row["case_id"],
                    "selected": mem["id"] in context_ids,
                })
    graph_only = sorted(set(graph_seen) - set(vector_seen))
    assert len(graph_only) == 6
    vector_db = readonly(VECTOR_DB)
    history_db = readonly(HISTORY_DB)
    cache_names = ("structured_cache.json", "topic_cache.json", "memory_embeddings.json")
    caches = {name: json.loads((ROOT / "data/graph_zh" / name).read_text()) for name in cache_names}
    rows = []
    for mid in graph_only:
        eid = vector_db.execute("select id from embeddings where embedding_id=?", (mid,)).fetchone()
        meta = {}
        if eid:
            for key, value in vector_db.execute(
                "select key,string_value from embedding_metadata where id=? and key in ('data','session_id','session_date','user_id')", (eid[0],)
            ):
                meta[key] = value
        history = [
            {"event": event, "created_at": created, "updated_at": updated, "is_deleted": deleted}
            for event, created, updated, deleted in history_db.execute(
                "select event,created_at,updated_at,is_deleted from history where memory_id=? order by coalesce(updated_at,created_at)", (mid,)
            )
        ]
        per_run = Counter(o["run"] for o in occurrences[mid])
        selected_per_run = Counter(o["run"] for o in occurrences[mid] if o["selected"])
        rows.append({
            "id": mid, "content": graph_seen[mid],
            "archived_graph_candidate_occurrences": len(occurrences[mid]),
            "archived_final_context_occurrences": sum(o["selected"] for o in occurrences[mid]),
            "candidate_occurrences_by_run": dict(per_run),
            "selected_occurrences_by_run": dict(selected_per_run),
            "current_vector_present": bool(eid),
            "current_vector_metadata": meta,
            "current_vector_content_equals_archived_graph": meta.get("data") == graph_seen[mid] if eid else None,
            "current_graph_cache_presence": {name: mid in cache for name, cache in caches.items()},
            "memory_history": history,
        })
    raw = json.loads(SOURCE.read_text())
    conv48 = next(c["conversation"] for c in raw if c["sample_id"] == "conv-48")
    evidence = [
        {"dia_id": t["dia_id"], "speaker": t["speaker"], "text": t["text"]}
        for t in conv48["session_8"] if t["dia_id"] in ("D8:14", "D8:16")
    ]
    deleted = next(r for r in rows if r["id"] == "bae6d454-5c37-4d97-9258-5b385204bf5e")
    assert not deleted["current_vector_present"]
    assert [h["event"] for h in deleted["memory_history"]] == ["ADD", "DELETE"]
    prior = json.loads(PRIOR_AUDIT.read_text())
    prior_unmatched = Counter(
        mid for case in prior["cases"] for mid in case["selected_graph_ids"]
        if mid not in vector_seen
    )
    assert sum(prior_unmatched.values()) == 8
    assert set(prior_unmatched) <= set(graph_only)
    result = {
        "scope": "Six IDs in archived Chinese graph candidate union but absent from archived vector retrieval union; current SQLite opened read-only",
        "counts": {"graph_only_ids": len(rows), "current_vector_present": sum(r["current_vector_present"] for r in rows),
                   "current_vector_absent": sum(not r["current_vector_present"] for r in rows),
                   "candidate_occurrences": sum(r["archived_graph_candidate_occurrences"] for r in rows),
                   "final_context_occurrences": sum(r["archived_final_context_occurrences"] for r in rows)},
        "ids": rows,
        "prior_graph_context_audit_unmatched_selected_instances_by_id": dict(prior_unmatched),
        "deleted_id_source": {"conversation": "conv-48", "session_id": "session_8", "session_date": conv48["session_8_date_time"], "turns": evidence},
        "limits": ["Absence from archived vector retrieval is not absence from the historical vector store",
                   "Current vector-store state is later than the archived runs; history event times must be compared with run times",
                   "Source turns confirm the deleted memory's core facts but do not prove every wording nuance or answer usefulness"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
