"""Frozen Chinese temporal answer POC: old vs parsed chronological order."""

import hashlib
import json
from pathlib import Path

from schema_rsi.benchmarks.base import parse_session_date
from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import (
    ANSWER_GENERATION_PROMPT, build_official_answer_prompt, format_memories_official,
)
from schema_rsi.llm.chat import make_chat_client
from schema_rsi.memory.base import MemoryRecord

from m2_probe_current_zh_stratified import signature


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results/analysis/m2_current_zh_stratified_retrieval_20260927.json"
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_chronological_order_answer_poc_20260927.json"
CASE_IDS = ["locomo_conv-42_qa20", "locomo_conv-48_qa2",
            "locomo_conv-41_qa59", "locomo_conv-50_qa26"]
ORDER = ["legacy", "chronological", "chronological", "legacy"]


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def memory_lines(records):
    return {rec.id: format_memories_official([rec]).splitlines()[-1] for rec in records}


def legacy_prompt(question, records, reference_date):
    lines = memory_lines(records)
    legacy = sorted(records, key=lambda rec: str(rec.metadata.get("session_date") or
                                                  rec.metadata.get("created_at") or ""))
    formatted = "\n".join([
        "The following memories are presented in chronological order (oldest to newest).",
        "", *(lines[rec.id] for rec in legacy),
    ])
    return ANSWER_GENERATION_PROMPT.format(
        memories=formatted, question=question, reference_date=reference_date or "2023")


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite prior calls: {OUT}")
    retrieval = json.loads(RETRIEVAL.read_text())
    assert retrieval["logical_store_unchanged"]
    before = signature()
    assert before == retrieval["logical_store_after"], "current vector memory differs from frozen retrieval"
    cases = {c["case_id"]: c for c in retrieval["cases"]}
    assert set(CASE_IDS) <= cases.keys()
    dataset = {s["sample_id"]: s for s in json.loads(DATASET.read_text())}
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    assert settings.evaluation.get("answer_prompt_preset") == "official"
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    result = {
        "scope": "Four previously selected/source-audited Chinese temporal questions; frozen no-graph top15; only context date order changes; ABBA; no judge",
        "source_retrieval": str(RETRIEVAL.relative_to(ROOT)),
        "source_retrieval_sha256": hashlib.sha256(RETRIEVAL.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "case_ids": CASE_IDS, "order_each_case": ORDER,
        "model": settings.llm.model, "temperature": 0.0,
        "max_answer_calls": len(CASE_IDS) * len(ORDER),
        "logical_store_before": before, "cases": [],
        "limits": ["A fixed four-question mechanism sample, not an accuracy estimate",
                   "Only date order changes; no graph candidate, date recovery or rerank change",
                   "No judge calls; source-based manual labels follow separately"],
    }
    for case_id in CASE_IDS:
        case = cases[case_id]
        conv = dataset[case["conversation_id"]]["conversation"]
        qa_index = int(case_id.rsplit("_qa", 1)[1])
        qa = dataset[case["conversation_id"]]["qa"][qa_index]
        dates = sorted((int(k.split("_")[1]), v) for k, v in conv.items()
                       if k.startswith("session_") and k.endswith("_date_time"))
        reference_date = parse_session_date(dates[-1][1]) if dates else None
        memories = [MemoryRecord(m["id"], m["content"],
                                 {k: m.get(k) for k in ("session_date", "session_id", "user_id")})
                    for m in case["reranked"]]
        assert len(memories) == 15
        new_prompt = build_official_answer_prompt(case["question"], memories, reference_date)
        old_prompt = legacy_prompt(case["question"], memories, reference_date)
        assert old_prompt != new_prompt
        lines = memory_lines(memories)
        assert all(old_prompt.count(line) == new_prompt.count(line) for line in lines.values())
        old_ids = [rec.id for rec in sorted(memories, key=lambda rec: rec.metadata["session_date"])]
        new_ids = [rec.id for rec in sorted(memories,
                    key=lambda rec: parse_session_date(rec.metadata["session_date"]))]
        row = {
            "case_id": case_id, "question": case["question"], "gold": qa.get("answer"),
            "reference_date": reference_date,
            "context_ids": [m.id for m in memories],
            "context_sha256": sha(json.dumps([(m.id, m.content, m.metadata) for m in memories], ensure_ascii=False)),
            "ordered_ids": {"legacy": old_ids, "chronological": new_ids},
            "prompt_sha256": {"legacy": sha(old_prompt), "chronological": sha(new_prompt)},
            "calls": [],
        }
        result["cases"].append(row)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        for arm in ORDER:
            if arm == "chronological":
                answer, info = answerer.answer(case["question"], memories, temperature=0.0,
                                               reference_date=reference_date, benchmark="locomo")
            else:
                answer, usage = client.complete(
                    system="You are answering a question using retrieved memories from past conversations.",
                    user=old_prompt, max_tokens=1500, temperature=0.0,
                )
                if "ANSWER:" in answer:
                    answer = answer.split("ANSWER:")[-1].strip() or answer.strip()
                info = {"usage": usage}
            row["calls"].append({"arm": arm, "answer": answer, **info})
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(case_id, arm, len(row["calls"]), answer[:150].replace("\n", " "), flush=True)
    assert sum(len(c["calls"]) for c in result["cases"]) == result["max_answer_calls"]
    result["logical_store_after"] = signature()
    result["logical_store_unchanged"] = before == result["logical_store_after"]
    result["completed"] = True
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("logical_store_unchanged", result["logical_store_unchanged"])


if __name__ == "__main__":
    main()
