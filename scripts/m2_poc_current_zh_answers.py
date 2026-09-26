#!/usr/bin/env python3
"""Five answer-only calls on frozen current Chinese no-graph top-15 contexts."""

import hashlib
import json
from pathlib import Path

from schema_rsi.benchmarks.base import parse_session_date
from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import build_official_answer_prompt
from schema_rsi.llm.chat import make_chat_client
from schema_rsi.memory.base import MemoryRecord


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json"
OUT = ROOT / "results/analysis/m2_current_zh_answer_poc_20260926.json"
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
VECTOR_DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"


def db_hash():
    digest = hashlib.sha256()
    with VECTOR_DB.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite prior calls: {OUT}")
    retrieval = json.loads(RETRIEVAL.read_text())
    before = db_hash()
    assert before == retrieval["vector_db_sha256_after"], "vector DB changed since retrieval probe"
    dataset = {row["sample_id"]: row for row in json.loads(DATASET.read_text())}
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    result = {
        "scope": "Five current Chinese no-graph answer calls on frozen reranked top-15, no judge calls",
        "source_retrieval": str(RETRIEVAL.relative_to(ROOT)),
        "model": settings.llm.model,
        "temperature": 0.0,
        "max_calls": 5,
        "vector_db_sha256_before": before,
        "cases": [],
        "limits": ["Five retrospective mechanism cases do not estimate overall accuracy", "Single answer sample per case; generation and judge noise unmeasured", "Current graph not rebuilt or evaluated"],
    }
    for case in retrieval["cases"]:
        conv_id = case["case_id"].removeprefix("locomo_").split("_qa")[0]
        conversation = dataset[conv_id]["conversation"]
        sessions = sorted(
            (int(k.split("_")[1]), v) for k, v in conversation.items()
            if k.startswith("session_") and k.endswith("_date_time")
        )
        reference_date = parse_session_date(sessions[-1][1]) if sessions else None
        memories = [MemoryRecord(
            id=m["id"], content=m["content"],
            metadata={"session_date": m["session_date"], "session_id": m["session_id"], "user_id": m["user_id"]},
        ) for m in case["reranked"]]
        assert len(memories) == 15
        prompt = build_official_answer_prompt(case["question"], memories, reference_date)
        answer, info = answerer.answer(
            case["question"], memories, temperature=0.0,
            reference_date=reference_date, benchmark="locomo",
        )
        result["cases"].append({
            "case_id": case["case_id"], "question": case["question"], "gold": case["gold"],
            "context_ids": [m.id for m in memories],
            "reference_date": reference_date,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "answer": answer, "usage": info["usage"], "answer_latency_s": info["answer_latency_s"],
        })
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(case["case_id"], answer[:180].replace("\n", " "), flush=True)
    result["vector_db_sha256_after"] = db_hash()
    result["vector_db_unchanged"] = result["vector_db_sha256_before"] == result["vector_db_sha256_after"]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}; vector_db_unchanged={result['vector_db_unchanged']}")


if __name__ == "__main__":
    main()
