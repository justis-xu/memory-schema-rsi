#!/usr/bin/env python3
"""Preselect Chinese LoCoMo cases by question metadata, then freeze no-graph retrieval."""

import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.llm.rerank import RerankClient
from schema_rsi.memory import Mem0Backend


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
MANIFEST = ROOT / "results/analysis/m2_current_zh_stratified_manifest_20260927.json"
OUT = ROOT / "results/analysis/m2_current_zh_stratified_retrieval_20260927.json"
EXCLUDE_FILES = [
    ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json",
    ROOT / "results/analysis/m2_multi_entity_cohort_retrieval_20260927.json",
]
SEED = "current-zh-stratified-20260927-v1"
PER_CATEGORY = 4
VECTOR_K, FINAL_K = 45, 15


def signature():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    h = hashlib.sha256()
    n = 0
    for row in con.execute("""
        select e.embedding_id,
          max(case when m.key='data' then m.string_value end),
          max(case when m.key='user_id' then m.string_value end),
          max(case when m.key='session_id' then m.string_value end),
          max(case when m.key='session_date' then m.string_value end)
        from embeddings e left join embedding_metadata m on m.id=e.id
        group by e.id order by e.embedding_id
    """):
        h.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        n += 1
    return {"embedding_count": n, "logical_sha256": h.hexdigest()}


def select():
    dataset = json.loads(DATASET.read_text())
    excluded = {c["case_id"] for path in EXCLUDE_FILES
                for c in json.loads(path.read_text())["cases"]}
    pools = defaultdict(list)
    for sample in dataset:
        for index, qa in enumerate(sample["qa"]):
            category = int(qa["category"])
            case_id = f"locomo_{sample['sample_id']}_qa{index}"
            if case_id in excluded:
                continue
            key = hashlib.sha256(f"{SEED}|{case_id}|{qa['question']}".encode()).hexdigest()
            pools[category].append({"case_id": case_id, "conversation_id": sample["sample_id"],
                                    "qa_index": index, "category": category,
                                    "question": qa["question"], "selection_hash": key})
    chosen = []
    for category in range(1, 6):
        used = set()
        for row in sorted(pools[category], key=lambda r: r["selection_hash"]):
            if row["conversation_id"] in used:
                continue
            chosen.append(row)
            used.add(row["conversation_id"])
            if len(used) == PER_CATEGORY:
                break
        assert len(used) == PER_CATEGORY
    return {
        "selection_rule": "Four smallest SHA256(seed|case_id|question) per category, with distinct conversations; exclude prior 5+8 probe cases; no gold/evidence read",
        "seed": SEED, "per_category": PER_CATEGORY, "selection_count": len(chosen),
        "excluded_case_ids": sorted(excluded), "cases": chosen,
    }


def serial(memory, rank, score=None):
    return {"rank": rank, "id": memory.id, "content": memory.content,
            "user_id": memory.metadata.get("user_id"),
            "session_id": memory.metadata.get("session_id"),
            "session_date": memory.metadata.get("session_date"),
            "rerank_score": score}


def run():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    manifest = json.loads(MANIFEST.read_text())
    assert manifest == select(), "manifest no longer matches source selection"
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    backend = Mem0Backend(settings)
    reranker = RerankClient(settings.rerank.base_url, settings.rerank.api_key, settings.rerank.model)
    result = {
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "scope": "Twenty preselected Chinese LoCoMo questions, no-graph retrieval only",
        "vector_k": VECTOR_K, "final_k": FINAL_K,
        "embedding_model": settings.embedding.model, "rerank_model": settings.rerank.model,
        "logical_store_before": signature(), "cases": [],
        "limits": ["Four per category and one per conversation within category; exploratory, not a population estimate", "Retrieval candidate presence is not answer correctness", "Adversarial category may have no supporting memory by design", "No answer or judge calls"],
    }
    for meta in manifest["cases"]:
        uid = f"zhfull:locomo:{meta['conversation_id']}"
        vector = backend.search_memory(meta["question"], user_id=uid, top_k=VECTOR_K)
        assert all(m.metadata.get("user_id") == uid for m in vector)
        scores = reranker.rerank(meta["question"], [m.content for m in vector], top_n=FINAL_K)
        final = [(vector[s["index"]], s["score"]) for s in scores
                 if 0 <= s["index"] < len(vector)][:FINAL_K]
        result["cases"].append({**meta, "user_id": uid,
                                "vector": [serial(m, i) for i, m in enumerate(vector, 1)],
                                "reranked": [serial(m, i, score) for i, (m, score) in enumerate(final, 1)],
                                "rerank_endpoint_style": reranker.last_endpoint})
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(meta["case_id"], len(vector), len(final), flush=True)
    result["logical_store_after"] = signature()
    result["logical_store_unchanged"] = result["logical_store_before"] == result["logical_store_after"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("logical_store_unchanged", result["logical_store_unchanged"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--manifest-only"]:
        if MANIFEST.exists():
            raise SystemExit(f"refusing to overwrite: {MANIFEST}")
        MANIFEST.write_text(json.dumps(select(), ensure_ascii=False, indent=2) + "\n")
        print(f"selected {PER_CATEGORY * 5} cases in {MANIFEST}")
    elif sys.argv[1:] == ["--run"]:
        run()
    else:
        raise SystemExit("usage: --manifest-only | --run")
