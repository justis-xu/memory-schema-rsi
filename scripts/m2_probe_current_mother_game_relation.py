"""One-question current Chinese retrieval probe for a source-supported relation."""

import hashlib
import json
import sqlite3
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.llm.rerank import RerankClient


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUT = ROOT / "results/analysis/m2_current_mother_game_relation_20260929.json"
USER = "zhfull:locomo:conv-48"
TARGET = "f1455645-db16-42df-a737-cb52619de275"
CASE = "locomo_conv-48_qa172"


def logical_signature():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = conn.execute("""SELECT e.embedding_id,
            MAX(CASE WHEN m.key='data' THEN m.string_value END),
            MAX(CASE WHEN m.key='user_id' THEN m.string_value END),
            MAX(CASE WHEN m.key='session_id' THEN m.string_value END),
            MAX(CASE WHEN m.key='session_date' THEN m.string_value END)
            FROM embeddings e LEFT JOIN embedding_metadata m ON e.id=m.id
            GROUP BY e.id ORDER BY e.embedding_id""").fetchall()
    finally:
        conn.close()
    return {"embedding_count": len(rows),
            "id_content_user_session_sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False,
                separators=(",", ":")).encode()).hexdigest()}, rows


def rank(ids, target):
    return ids.index(target) + 1 if target in ids else None


def save(report):
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    source = next(s for s in json.loads(DATA.read_text()) if s["sample_id"] == "conv-48")
    qa = source["qa"][172]
    turns = [t for t in source["conversation"]["session_24"]
             if t["dia_id"] in ("D24:8", "D24:9", "D24:10")]
    assert [t["dia_id"] for t in turns] == ["D24:8", "D24:9", "D24:10"]
    before, rows = logical_signature()
    target_rows = [r for r in rows if r[0] == TARGET]
    assert len(target_rows) == 1 and target_rows[0][2] == USER
    assert "妈妈" in target_rows[0][1] and "Monster Hunter: World" in target_rows[0][1]
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    report = {
        "scope": "Current pure-Chinese one-question vector top45 and rerank top15; no Graph, answer, or judge.",
        "case_id": CASE, "question": qa["question"], "gold_for_review_only": qa["answer"],
        "source_dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "source_turns": [{k: t.get(k) for k in ("dia_id", "speaker", "text", "blip_caption", "query")}
                         for t in turns],
        "user_id": USER, "target_memory_id": TARGET,
        "target_current_memory": {"id": TARGET, "content": target_rows[0][1],
                                  "session_id": target_rows[0][3], "session_date": target_rows[0][4]},
        "logical_source_before": before, "vector_k": 45, "final_k": 15,
        "max_embedding_calls": 1, "max_logical_rerank_calls": 1,
        "max_rerank_http_attempts": 2,
        "embedding_model": settings.embedding.model, "rerank_model": settings.rerank.model,
        "calls": {"embedding": 0, "logical_rerank": 0},
        "limits": [
            "Source relation uses D24:8-10; Wii appears in the D24:8 image caption, not in the quoted text.",
            "Candidate presence does not establish an answer or a graph/Jev benefit.",
            "The current source is not an immutable historical run snapshot.",
        ],
    }
    save(report)
    try:
        import chromadb
        from openai import OpenAI

        chroma = chromadb.PersistentClient(path=str(DB.parent))
        collection = chroma.get_collection(settings.mem0["collection_name"])
        embed = OpenAI(base_url=settings.embedding.base_url,
                       api_key=settings.embedding.api_key, timeout=20, max_retries=0)
        report["calls"]["embedding"] = 1
        vector = embed.embeddings.create(model=settings.embedding.model,
                                         input=[qa["question"]]).data[0].embedding
        hits = collection.query(query_embeddings=[vector], n_results=45,
                                where={"user_id": USER}, include=["metadatas", "distances"])
        ids, metas, distances = hits["ids"][0], hits["metadatas"][0], hits["distances"][0]
        assert len(ids) == len(metas) == len(distances) == 45
        assert all(m.get("user_id") == USER and m.get("data") for m in metas)
        report["vector_top45"] = [
            {"rank": n, "id": mid, "content": meta["data"],
             "session_id": meta.get("session_id"), "session_date": meta.get("session_date"),
             "distance": distance}
            for n, (mid, meta, distance) in enumerate(zip(ids, metas, distances), 1)
        ]
        report["target_vector_rank"] = rank(ids, TARGET)
        save(report)

        reranker = RerankClient(settings.rerank.base_url, settings.rerank.api_key,
                                settings.rerank.model, timeout=30)
        report["calls"]["logical_rerank"] = 1
        scored = reranker.rerank(qa["question"], [m["data"] for m in metas], top_n=15)
        indices = [item["index"] for item in scored]
        assert len(indices) == len(set(indices)) == 15
        assert all(0 <= i < 45 for i in indices)
        report["rerank_top15"] = [
            {**report["vector_top45"][item["index"]], "rank": n,
             "vector_rank": item["index"] + 1, "rerank_score": item["score"]}
            for n, item in enumerate(scored, 1)
        ]
        report["rerank_endpoint_style"] = reranker.last_endpoint
        report["target_final_rank"] = rank([m["id"] for m in report["rerank_top15"]], TARGET)
        report["stopped_after_failure"] = False
        print("target", report["target_vector_rank"], report["target_final_rank"], flush=True)
    except Exception as exc:
        report["stopped_after_failure"] = True
        report["error_type"] = type(exc).__name__
        report["http_status"] = getattr(exc, "status_code", None)
        print("stopped", report["error_type"], report["http_status"], flush=True)
    finally:
        after, _ = logical_signature()
        report["logical_source_after"] = after
        report["logical_source_unchanged"] = before == after
        save(report)
        print("logical source unchanged", report["logical_source_unchanged"], flush=True)


if __name__ == "__main__":
    main()
