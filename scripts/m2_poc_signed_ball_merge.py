#!/usr/bin/env python3
"""Four ABBA answer calls after one current Chinese signed-ball memory swap."""

import hashlib
import json
import sqlite3
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import build_official_answer_prompt
from schema_rsi.llm.chat import make_chat_client
from schema_rsi.memory.base import MemoryRecord


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json"
PATHS = ROOT / "results/analysis/m2_signed_ball_paths_20260926.json"
PRIOR_ANSWER = ROOT / "results/analysis/m2_current_zh_answer_poc_20260926.json"
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUT = ROOT / "results/analysis/m2_signed_ball_merge_poc_20260926.json"
CASE_ID = "locomo_conv-43_qa25"
JOHN_ID = "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"
ORDER = ["base", "merge", "merge", "base"]


def file_hash():
    return hashlib.sha256(DB.read_bytes()).hexdigest()


def current_memory_metadata(memory_id):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    eid = con.execute("select id from embeddings where embedding_id=?", (memory_id,)).fetchone()
    assert eid is not None
    return {key: value for key, value in con.execute(
        "select key,string_value from embedding_metadata where id=? and key in ('data','session_id','session_date','user_id')", (eid[0],)
    )}


def to_record(row):
    return MemoryRecord(
        id=row["id"], content=row["content"],
        metadata={"session_id": row.get("session_id"), "session_date": row.get("session_date"), "user_id": row.get("user_id")},
    )


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite prior calls: {OUT}")
    retrieval = json.loads(RETRIEVAL.read_text())
    paths = json.loads(PATHS.read_text())
    prior = json.loads(PRIOR_ANSWER.read_text())
    case = next(c for c in retrieval["cases"] if c["case_id"] == CASE_ID)
    prior_case = next(c for c in prior["cases"] if c["case_id"] == CASE_ID)
    base = [dict(m) for m in case["reranked"]]
    assert [m["id"] for m in base] == prior_case["context_ids"]
    john_query = next(q for q in paths["queries"] if q["query"].startswith("约翰"))
    candidate = next(m for m in john_query["reranked"] if m["id"] == JOHN_ID)
    meta = current_memory_metadata(JOHN_ID)
    assert meta["data"] == candidate["content"] and meta["user_id"] == case["user_id"]
    merge = [dict(m) for m in base]
    removed = merge[-1]
    merge[-1] = dict(candidate, rank=15, session_date=meta["session_date"], user_id=meta["user_id"])
    assert sum(a["id"] != b["id"] for a, b in zip(base, merge)) == 1
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    contexts = {"base": [to_record(m) for m in base], "merge": [to_record(m) for m in merge]}
    prompts = {arm: build_official_answer_prompt(case["question"], records, prior_case["reference_date"])
               for arm, records in contexts.items()}
    result = {
        "scope": "One current Chinese two-person question; fixed 15-slot one-memory swap; ABBA, max four answer calls, no judge",
        "source_retrieval": str(RETRIEVAL.relative_to(ROOT)),
        "source_person_split": str(PATHS.relative_to(ROOT)),
        "question": case["question"], "gold": case["gold"],
        "model": settings.llm.model, "temperature": 0.0, "order": ORDER,
        "removed": {"id": removed["id"], "content": removed["content"]},
        "added": {"id": candidate["id"], "content": candidate["content"], "session_date": meta["session_date"]},
        "base_ids": [m.id for m in contexts["base"]],
        "merge_ids": [m.id for m in contexts["merge"]],
        "prompt_sha256": {arm: hashlib.sha256(prompt.encode()).hexdigest() for arm, prompt in prompts.items()},
        "vector_db_sha256_before": file_hash(),
        "calls": [],
        "limits": ["The John memory was chosen with source-aware audit after non-oracle retrieval; automatic merge selection is not evaluated", "One question and two samples per arm cannot estimate overall benefit", "Extra person-split retrieval cost is not included in four answer calls"],
    }
    for index, arm in enumerate(ORDER, 1):
        answer, info = answerer.answer(
            case["question"], contexts[arm], temperature=0.0,
            reference_date=prior_case["reference_date"], benchmark="locomo",
        )
        result["calls"].append({"index": index, "arm": arm, "answer": answer, "usage": info["usage"]})
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(index, arm, answer[:170].replace("\n", " "), flush=True)
    result["vector_db_sha256_after"] = file_hash()
    result["vector_db_unchanged"] = result["vector_db_sha256_before"] == result["vector_db_sha256_after"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}; vector_db_unchanged={result['vector_db_unchanged']}")


if __name__ == "__main__":
    main()
