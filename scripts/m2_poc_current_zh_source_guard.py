#!/usr/bin/env python3
"""Frozen Chinese answer POC: official prompt versus source-boundary addendum."""

import hashlib
import json
from pathlib import Path

from schema_rsi.benchmarks.base import parse_session_date
from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import build_official_answer_prompt
from schema_rsi.llm.chat import make_chat_client
from schema_rsi.memory.base import MemoryRecord

from m2_probe_current_zh_stratified import signature


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results/analysis/m2_current_zh_stratified_retrieval_20260927.json"
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_current_zh_source_guard_poc_20260927.json"
CASE_IDS = [
    "locomo_conv-48_qa20",   # source says played / planned / acquired
    "locomo_conv-48_qa233",  # speaker swap
    "locomo_conv-42_qa215",  # false numb-mind premise
    "locomo_conv-30_qa85",   # participation speaker swap
    "locomo_conv-26_qa157",  # necklace owner swap
    "locomo_conv-26_qa111",  # clean direct-answer harm control
]
ORDER = ["official", "guard", "guard", "official"]
ADDENDUM = """

## SOURCE BOUNDARY (overrides any instruction above that would force an unsupported answer)
Before answering, verify that the memories support the exact person, action, time, and whether it happened, was only planned, or was only acquired. Do not transfer an action or possession from one speaker to another. Do not turn a plan into a completed event or a nearby fact into proof of an unstated premise. If the requested fact or premise is not supported, say briefly that it cannot be determined from the memories; you may name the supported fact separately when useful. When the exact requested fact is supported, answer it directly and specifically. Do not refuse merely because the memories are imperfect.
"""


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def records(rows):
    return [MemoryRecord(
        id=m["id"], content=m["content"],
        metadata={"session_date": m["session_date"], "session_id": m["session_id"], "user_id": m["user_id"]},
    ) for m in rows]


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite prior calls: {OUT}")
    retrieval = json.loads(RETRIEVAL.read_text())
    assert retrieval["logical_store_unchanged"]
    before = signature()
    assert before == retrieval["logical_store_after"], "current vector memory differs from frozen retrieval"
    cases = {c["case_id"]: c for c in retrieval["cases"]}
    assert len(cases) == 20 and len(CASE_IDS) == len(set(CASE_IDS)) == 6
    assert set(CASE_IDS) <= cases.keys()
    dataset = {s["sample_id"]: s for s in json.loads(DATASET.read_text())}
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    assert settings.evaluation.get("answer_prompt_preset") == "official"
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    result = {
        "scope": "Six source-audited Chinese cases; frozen current no-graph top 15; official vs source-boundary addendum; ABBA; no judge",
        "source_retrieval": str(RETRIEVAL.relative_to(ROOT)),
        "case_ids": CASE_IDS, "order_each_case": ORDER,
        "model": settings.llm.model, "temperature": 0.0,
        "max_answer_calls": len(CASE_IDS) * len(ORDER),
        "guard_addendum": ADDENDUM,
        "logical_store_before": before, "cases": [],
        "limits": ["Purposefully chosen source-audited cases, not an accuracy estimate", "Prompt addendum is a candidate mechanism, not a deployed change", "No judge calls; source-level manual labels follow separately"],
    }
    for case_id in CASE_IDS:
        case = cases[case_id]
        conv = dataset[case["conversation_id"]]["conversation"]
        dates = sorted((int(k.split("_")[1]), v) for k, v in conv.items()
                       if k.startswith("session_") and k.endswith("_date_time"))
        reference_date = parse_session_date(dates[-1][1]) if dates else None
        memories = records(case["reranked"])
        assert len(memories) == 15
        base_prompt = build_official_answer_prompt(case["question"], memories, reference_date)
        row = {
            "case_id": case_id, "question": case["question"],
            "reference_date": reference_date,
            "context_ids": [m.id for m in memories],
            "context_sha256": sha(json.dumps([(m.id, m.content, m.metadata) for m in memories], ensure_ascii=False)),
            "prompt_sha256": {"official": sha(base_prompt), "guard": sha(base_prompt + ADDENDUM)},
            "calls": [],
        }
        result["cases"].append(row)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        for arm in ORDER:
            if arm == "official":
                answer, info = answerer.answer(case["question"], memories, temperature=0.0,
                                               reference_date=reference_date, benchmark="locomo")
            else:
                answer, usage = client.complete(
                    system="You are answering a question using retrieved memories from past conversations.",
                    user=base_prompt + ADDENDUM, max_tokens=1500, temperature=0.0,
                )
                if "ANSWER:" in answer:
                    answer = answer.split("ANSWER:")[-1].strip() or answer.strip()
                info = {"usage": usage}
            row["calls"].append({"arm": arm, "answer": answer, **info})
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(case_id, arm, len(row["calls"]), answer[:150].replace("\n", " "), flush=True)
    assert sum(len(c["calls"]) for c in result["cases"]) == result["max_answer_calls"]
    result["logical_store_after"] = signature()
    result["logical_store_unchanged"] = result["logical_store_before"] == result["logical_store_after"]
    result["completed"] = True
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("logical_store_unchanged", result["logical_store_unchanged"])


if __name__ == "__main__":
    main()
