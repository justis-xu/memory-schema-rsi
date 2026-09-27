#!/usr/bin/env python3
"""Frozen Chinese two-person cohort: ABBA answers for original vs split merge."""

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
RETRIEVAL = ROOT / "results/analysis/m2_multi_entity_cohort_retrieval_20260927.json"
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_multi_entity_cohort_answer_poc_20260927.json"
ORDER = ["base", "merge", "merge", "base"]


def records(rows):
    return [MemoryRecord(id=m["id"], content=m["content"], metadata={
        "session_date": m["session_date"], "session_id": m["session_id"], "user_id": m["user_id"],
    }) for m in rows]


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite prior calls: {OUT}")
    retrieval = json.loads(RETRIEVAL.read_text())
    assert retrieval["selection_count"] == len(retrieval["cases"]) == 8
    assert retrieval["logical_store_unchanged"]
    dataset = {s["sample_id"]: s for s in json.loads(DATASET.read_text())}
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    result = {
        "scope": "All eight preselected Chinese joint-person category-1 questions; frozen original/merge contexts; ABBA each; no judge",
        "source_retrieval": str(RETRIEVAL.relative_to(ROOT)),
        "model": settings.llm.model, "temperature": 0.0,
        "max_answer_calls": 32, "planned_order_each_case": ORDER,
        "cases": [],
        "limits": ["Eight pattern-selected cases do not estimate overall score", "Two samples per arm only partially measure answer noise", "Merged context costs two extra vector searches and two extra reranks per case", "Gold is stored for later audit but not used in retrieval or merge"],
    }
    for case in retrieval["cases"]:
        conv = dataset[case["conversation_id"]]["conversation"]
        sessions = sorted((int(k.split("_")[1]), v) for k, v in conv.items()
                          if k.startswith("session_") and k.endswith("_date_time"))
        reference_date = parse_session_date(sessions[-1][1]) if sessions else None
        qa_index = int(case["case_id"].split("_qa")[-1])
        qa = dataset[case["conversation_id"]]["qa"][qa_index]
        assert qa["question"] == case["question"]
        contexts = {
            "base": records(case["arms"]["original"]["reranked"]),
            "merge": records(case["merged_context"]),
        }
        assert all(len(v) == 15 for v in contexts.values())
        row = {
            "case_id": case["case_id"], "question": case["question"], "gold": qa["answer"],
            "evidence": qa.get("evidence"), "reference_date": reference_date,
            "context_ids": {arm: [m.id for m in rec] for arm, rec in contexts.items()},
            "prompt_sha256": {arm: hashlib.sha256(build_official_answer_prompt(case["question"], rec, reference_date).encode()).hexdigest()
                              for arm, rec in contexts.items()},
            "calls": [],
        }
        result["cases"].append(row)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        for arm in ORDER:
            answer, info = answerer.answer(case["question"], contexts[arm], temperature=0.0,
                                           reference_date=reference_date, benchmark="locomo")
            row["calls"].append({"arm": arm, "answer": answer, "usage": info["usage"],
                                 "answer_latency_s": info["answer_latency_s"]})
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(case["case_id"], arm, len(row["calls"]), answer[:160].replace("\n", " "), flush=True)
    assert sum(len(c["calls"]) for c in result["cases"]) == 32
    result["completed"] = True
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
