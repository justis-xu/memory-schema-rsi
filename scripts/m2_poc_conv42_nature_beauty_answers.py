#!/usr/bin/env python3
"""Frozen answer check for current-memory vs source-turn visibility in one Chinese case."""

import argparse
import hashlib
import json
from pathlib import Path

from schema_rsi.benchmarks.base import parse_session_date
from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import build_official_answer_prompt
from schema_rsi.llm.chat import make_chat_client
from schema_rsi.memory.base import MemoryRecord

from m2_audit_zh_graph_cache_alignment import current_memories, signature


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
RETRIEVAL = ROOT / "results/analysis/m2_multi_entity_cohort_retrieval_20260927.json"
PRIOR = ROOT / "results/analysis/m2_multi_entity_cohort_answer_poc_20260927.json"
OUT = ROOT / "results/analysis/m2_conv42_nature_beauty_answer_poc_20260927.json"
CASE = "locomo_conv-42_qa71"
EXISTING_PREFIXES = ("e196eb2e", "56217cd3")
SOURCE_DIA_IDS = ("D11:9", "D28:23")
ORDER = ("base", "existing", "source", "source", "existing", "base")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rec(row):
    return MemoryRecord(id=row["id"], content=row["content"], metadata={
        "user_id": row["user_id"], "session_id": row["session_id"],
        "session_date": row["session_date"]})


def build():
    dataset = next(s for s in json.loads(DATASET.read_text()) if s["sample_id"] == "conv-42")
    cohort = json.loads(RETRIEVAL.read_text())
    case = next(c for c in cohort["cases"] if c["case_id"] == CASE)
    prior_case = next(c for c in json.loads(PRIOR.read_text())["cases"] if c["case_id"] == CASE)
    original = case["arms"]["original"]["reranked"]
    assert len(original) == 15 and case["question"] == prior_case["question"]
    current = current_memories()
    for row in original:
        assert current[row["id"]]["content"] == row["content"], row["id"]
    common = [rec(row) for row in original[:13]]
    base = common + [rec(row) for row in original[13:]]
    existing = []
    for prefix in EXISTING_PREFIXES:
        matches = [(mid, row) for mid, row in current.items() if mid.startswith(prefix)]
        assert len(matches) == 1
        mid, row = matches[0]
        assert row["user_id"] == case["user_id"]
        frozen = next((r for arm in case["arms"].values()
                       for r in arm["vector"] if r["id"] == mid), None)
        if frozen is not None:
            assert frozen["content"] == row["content"]
        session_date = dataset["conversation"][row["session_id"] + "_date_time"]
        existing.append(MemoryRecord(id=mid, content=row["content"], metadata={
            "user_id": row["user_id"], "session_id": row["session_id"],
            "session_date": session_date}))
    source = []
    for dia in SOURCE_DIA_IDS:
        matches = [(sid, row) for sid, rows in dataset["conversation"].items()
                   if sid.startswith("session_") and not sid.endswith("_date_time")
                   for row in rows if row["dia_id"] == dia]
        assert len(matches) == 1
        sid, turn = matches[0]
        source.append(MemoryRecord(id=f"source-turn:{dia}",
                                   content=f"{turn['speaker']}：{turn['text']}", metadata={
            "user_id": case["user_id"], "session_id": sid,
            "session_date": dataset["conversation"][sid + "_date_time"]}))
    contexts = {"base": base, "existing": common + existing, "source": common + source}
    assert all(len(v) == 15 for v in contexts.values())
    assert all([r.id for r in recs[:13]] == [r.id for r in common]
               for recs in contexts.values())
    dates = [parse_session_date(v) for k, v in dataset["conversation"].items()
             if k.startswith("session_") and k.endswith("_date_time")]
    reference_date = max(dates)
    assert str(reference_date) == str(prior_case["reference_date"])
    prompts = {name: build_official_answer_prompt(case["question"], recs, reference_date)
               for name, recs in contexts.items()}
    return case, contexts, reference_date, prompts, current, prior_case["prompt_sha256"]["base"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make up to six answer calls")
    args = parser.parse_args()
    case, contexts, reference_date, prompts, current, prior_base_prompt_sha = build()
    prepared = {
        "scope": "One source-audited Chinese question; original top13 fixed and last two slots controlled; source arm is an oracle upper bound, no judge",
        "case_id": CASE, "question": case["question"], "gold": "大自然",
        "source_dia_ids": SOURCE_DIA_IDS,
        "max_answer_calls": 6, "order": ORDER, "answer_calls": 0,
        "reference_date": str(reference_date),
        "current_memory_signature": signature(current),
        "context_ids": {name: [r.id for r in rows] for name, rows in contexts.items()},
        "context_last_two": {name: [{"id": r.id, "content": r.content,
                                     "metadata": r.metadata} for r in rows[-2:]]
                             for name, rows in contexts.items()},
        "prompt_sha256": {name: sha(prompt) for name, prompt in prompts.items()},
        "prior_base_prompt_sha256": prior_base_prompt_sha,
        "current_base_prompt_matches_prior": sha(prompts["base"]) == prior_base_prompt_sha,
        "prompt_chars": {name: len(prompt) for name, prompt in prompts.items()},
        "calls": [],
        "limits": ["Source arm inserts gold-evidence turns manually; it is an oracle upper bound, not an automatic memory extraction method.",
                   "Base vs other arms changes two contents at fixed slots; removal of old distractors is not separately identified.",
                   "Existing vs source changes two records, including exact relation wording; attribution to a single turn is not identified.",
                   "One question and two repeats per arm cannot estimate general answer benefit or noise distribution."],
    }
    if not args.run:
        if OUT.exists():
            raise SystemExit(f"refusing to overwrite: {OUT}")
        OUT.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n")
        print("prepared", CASE, prepared["prompt_chars"], "max calls", len(ORDER))
        return
    if not OUT.exists():
        raise SystemExit("prepare first")
    report = json.loads(OUT.read_text())
    for key in ("context_ids", "prompt_sha256", "current_memory_signature"):
        assert report[key] == prepared[key], key
    assert report["answer_calls"] == 0
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    report["model"] = settings.llm.model
    report["temperature"] = 0.0
    for index, arm in enumerate(ORDER, 1):
        try:
            answer, info = answerer.answer(case["question"], contexts[arm],
                                           temperature=0.0, reference_date=reference_date,
                                           benchmark="locomo")
            report["answer_calls"] += 1
            report["calls"].append({"index": index, "arm": arm, "answer": answer,
                                    "usage": info.get("usage")})
        except Exception as exc:
            report["answer_calls"] += 1
            report["calls"].append({"index": index, "arm": arm,
                                    "error_type": type(exc).__name__})
            report["stopped_early"] = True
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if report.get("stopped_early"):
            print("stopped after", report["answer_calls"], "call(s)")
            return
        print(index, arm, answer[:180].replace("\n", " "), flush=True)
    report["completed"] = True
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
