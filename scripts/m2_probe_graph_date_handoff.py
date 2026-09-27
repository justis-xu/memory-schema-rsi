"""Trace a current sourced graph memory date into the answer prompt without services."""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m2_replay_signed_ball_graph import LocalGraph  # noqa: E402
from schema_rsi.evaluation.pipeline import _graph_memory_record  # noqa: E402
from schema_rsi.evaluation.prompts_official import format_memories_official  # noqa: E402
from schema_rsi.evaluation.retriever import GraphRetriever  # noqa: E402
from schema_rsi.graph.builder import GraphBuilder  # noqa: E402
from schema_rsi.memory.base import MemoryRecord  # noqa: E402


DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
CACHE = ROOT / "data/graph_zh/structured_cache.json"
OUT = ROOT / "results/analysis/m2_graph_date_handoff_20260927.json"
TIM = "219b4c97-f2af-4734-a4f4-1d93ebe19f99"
JOHN = "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"


def db_hash():
    return hashlib.sha256(DB.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    before = db_hash()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("""
        select e.embedding_id, m.key, m.string_value
        from embeddings e join embedding_metadata m on m.id=e.id
        where e.embedding_id in (?, ?)
    """, (TIM, JOHN)).fetchall()
    con.close()
    metadata = {TIM: {}, JOHN: {}}
    for mid, key, value in rows:
        metadata[mid][key] = value
    assert all(metadata[mid].get("session_date") for mid in (TIM, JOHN))
    records = [MemoryRecord(mid, metadata[mid]["data"], metadata[mid]) for mid in (TIM, JOHN)]
    cache = json.loads(CACHE.read_text())
    graph = LocalGraph()
    GraphBuilder(graph).build_s1(records, {mid: cache[mid] for mid in (TIM, JOHN)})
    candidates = GraphRetriever(graph).retrieve_fused(
        [records[0]], exclude_ids={TIM}, per_node_limit=15, max_candidates=30,
        hops=[("Entity", "MENTIONS", "entity_id")])
    john = next(c for c in candidates if c["id"] == JOHN)
    answer_record = _graph_memory_record(john)
    formatted = format_memories_official([answer_record])
    assert answer_record.metadata["session_date"] == metadata[JOHN]["session_date"]
    assert "unknown date" not in formatted
    after = db_hash()
    assert before == after
    result = {
        "scope": "Two current conv-43 memories through production S1 builder, fused graph retriever, graph candidate conversion and official formatter in an isolated local store; no service/model calls",
        "current_db_sha256_before": before, "current_db_sha256_after": after,
        "tim_id": TIM, "john_id": JOHN,
        "john_source_session_id": metadata[JOHN]["session_id"],
        "john_source_session_date": metadata[JOHN]["session_date"],
        "john_graph_candidate_session_id": john["session_id"],
        "john_graph_candidate_session_date": john["session_date"],
        "john_answer_record_session_date": answer_record.metadata["session_date"],
        "formatted_date_prefix": formatted.splitlines()[-1].split(") ", 1)[0] + ")",
        "date_unknown_in_formatted_prompt": "unknown date" in formatted,
        "limits": ["Only a two-memory, source-checked path; this does not test real HugeGraph edge order, fused reranking or answer accuracy.",
                   "Graph vertices without a source session_date remain undated; this change does not infer a date from memory content."],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
