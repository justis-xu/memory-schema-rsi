#!/usr/bin/env python3
"""Preselected Chinese two-person cohort: original vs person-split retrieval.

Selection uses only question text, speaker names and category. It never reads
gold answers or evidence while choosing cases or composing subqueries.
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.llm.rerank import RerankClient
from schema_rsi.memory import Mem0Backend


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUT = ROOT / "results/analysis/m2_multi_entity_cohort_retrieval_20260927.json"
VECTOR_K, FINAL_K, PER_QUERY_QUOTA = 45, 15, 5


def logical_signature():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    h = hashlib.sha256()
    rows = con.execute("""
        select e.embedding_id,
          max(case when m.key='data' then m.string_value end),
          max(case when m.key='user_id' then m.string_value end),
          max(case when m.key='session_id' then m.string_value end),
          max(case when m.key='session_date' then m.string_value end)
        from embeddings e left join embedding_metadata m on m.id=e.id
        group by e.id order by e.embedding_id
    """)
    count = 0
    for row in rows:
        h.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        count += 1
    return {"embedding_count": count, "id_content_user_session_sha256": h.hexdigest()}


def select_cases(dataset):
    selected = []
    for sample in dataset:
        a, b = sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]
        pattern = re.compile(f"(?:{re.escape(a)}和{re.escape(b)}|{re.escape(b)}和{re.escape(a)})都")
        for index, qa in enumerate(sample["qa"]):
            if qa["category"] != 1:
                continue
            match = pattern.search(qa["question"])
            if not match:
                continue
            suffix = qa["question"][match.end():]
            prefix = qa["question"][:match.start()]
            queries = [prefix + person + suffix for person in (a, b)]
            selected.append({
                "case_id": f"locomo_{sample['sample_id']}_qa{index}",
                "conversation_id": sample["sample_id"],
                "question": qa["question"], "speaker_names": [a, b],
                "queries": [qa["question"], *queries],
            })
    return selected


def serial(record, rank, score=None):
    return {"rank": rank, "id": record.id, "content": record.content,
            "session_id": record.metadata.get("session_id"),
            "session_date": record.metadata.get("session_date"),
            "user_id": record.metadata.get("user_id"), "rerank_score": score}


def merge_context(arms):
    # Deterministic, no gold: protect five original positions, then give each
    # person five slots, deduplicate IDs and fill unused space from original.
    seen = set()
    merged = []
    for name, limit in [("original", 5), ("speaker_a", 5), ("speaker_b", 5), ("original", 15)]:
        for memory in arms[name]["reranked"][:limit]:
            if memory["id"] not in seen and len(merged) < 15:
                seen.add(memory["id"])
                merged.append({**memory, "rank": len(merged) + 1, "first_admitted_via": name})
    assert len(merged) == 15
    return merged


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    dataset = json.loads(DATASET.read_text())
    cohort = select_cases(dataset)
    assert len(cohort) == 8, [(row["case_id"], row["question"]) for row in cohort]
    before = logical_signature()
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    backend = Mem0Backend(settings)
    reranker = RerankClient(settings.rerank.base_url, settings.rerank.api_key, settings.rerank.model)
    result = {
        "scope": "All Chinese LoCoMo category-1 questions with an exact speaker-A-and-speaker-B-都 phrase; selection and subqueries do not use gold",
        "dataset": str(DATASET), "selection_count": len(cohort),
        "selection_rule": "category==1 and exact regex (speaker_a和speaker_b|speaker_b和speaker_a)都",
        "subquery_rule": "replace the joint phrase including 都 with each speaker name; keep prefix and suffix",
        "merge_rule": "first five original reranked, first five per speaker reranked, stable dedup, fill from original to 15",
        "vector_k": VECTOR_K, "final_k": FINAL_K, "per_query_quota": PER_QUERY_QUOTA,
        "embedding_model": settings.embedding.model, "rerank_model": settings.rerank.model,
        "logical_store_before": before, "cases": [],
        "limits": ["Eight text-pattern-selected questions, not all multi-entity questions", "Candidate admission is not answer quality or source support", "The query split uses known conversation speakers, as the dataset runner does", "No answer or judge calls"],
    }
    for meta in cohort:
        uid = f"zhfull:locomo:{meta['conversation_id']}"
        arms = {}
        for name, query in zip(("original", "speaker_a", "speaker_b"), meta["queries"]):
            vector = backend.search_memory(query, user_id=uid, top_k=VECTOR_K)
            assert all(m.metadata.get("user_id") == uid for m in vector)
            scored = reranker.rerank(query, [m.content for m in vector], top_n=FINAL_K)
            final = [vector[s["index"]] for s in scored if 0 <= s["index"] < len(vector)][:FINAL_K]
            arms[name] = {
                "query": query,
                "vector": [serial(m, i) for i, m in enumerate(vector, 1)],
                "reranked": [serial(m, i, scored[i-1]["score"]) for i, m in enumerate(final, 1)],
                "rerank_endpoint_style": reranker.last_endpoint,
            }
        merged = merge_context(arms)
        base_ids = {m["id"] for m in arms["original"]["reranked"]}
        merged_ids = {m["id"] for m in merged}
        result["cases"].append({
            **meta, "user_id": uid, "arms": arms, "merged_context": merged,
            "gained_ids": [m["id"] for m in merged if m["id"] not in base_ids],
            "displaced_ids": [m["id"] for m in arms["original"]["reranked"] if m["id"] not in merged_ids],
        })
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(meta["case_id"], 'gained',len(result["cases"][-1]["gained_ids"]),
              'displaced',len(result["cases"][-1]["displaced_ids"]),flush=True)
    result["logical_store_after"] = logical_signature()
    result["logical_store_unchanged"] = result["logical_store_before"] == result["logical_store_after"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}; logical_store_unchanged={result['logical_store_unchanged']}")


if __name__ == "__main__":
    main()
