#!/usr/bin/env python3
"""Run at most 12 answer calls on three frozen Chinese graph-swap cases."""

import hashlib
import json
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.evaluation.prompts_official import ANSWER_GENERATION_PROMPT
from schema_rsi.llm.chat import make_chat_client


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/analysis/m2_graph_fact_bridge_cases_20260926.json"
OUT = ROOT / "results/analysis/m2_frozen_graph_swaps_poc_20260926.json"

# One memory replacement per case; all other content and retrieval order fixed.
SWAPS = {
    "locomo_conv-47_qa96": {"replace_rank": 15, "graph_memory_id": "3cded448-5856-4da0-81c6-aa9c84cdf61d"},
    "locomo_conv-41_qa86": {"replace_rank": 15, "graph_memory_id": "f8681a3d-0e6a-4d52-b20e-871a78f9495c"},
    "locomo_conv-30_qa80": {"replace_rank": 14, "graph_memory_id": "72c1939a-c0be-463d-bb94-481e0779fa2a"},
}
ORDER = ["base", "swap", "swap", "base"]  # ABBA within each question
SYSTEM = "You are answering a question using retrieved memories from past conversations."


def prompt(question, context):
    # The historical official template is kept, but per-memory dates are omitted
    # for these non-temporal questions so candidate content is the only changed
    # prompt field. The order and all 15 other-line positions remain fixed.
    memories = "Memories in fixed retrieval order:\n" + "\n".join(
        f"{rank}. {memory['content']}" for rank, memory in enumerate(context, 1)
    )
    return ANSWER_GENERATION_PROMPT.format(
        reference_date="2023", memories=memories, question=question
    )


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite prior calls: {OUT}")
    packet = json.loads(PACKET.read_text())
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)  # preserve the 12-request cap
    result = {
        "scope": "Three retrospective Chinese cases; one slot swapped; two responses per arm; no judge calls",
        "model": settings.llm.model,
        "temperature": 0.0,
        "max_calls": len(SWAPS) * len(ORDER),
        "max_completion_tokens_per_call": 1200,
        "order_per_case": ORDER,
        "prompt_template": "official LoCoMo answer template; fixed retrieval-order numbered memories without date tags",
        "source_packet": str(PACKET.relative_to(ROOT)),
        "cases": [],
        "limits": [
            "Old mixed-language archived memories, not the current pure-Chinese memory store.",
            "The one-slot swap tests candidate content, not historical graph traversal, ranker, dates or overall net score.",
            "Two responses per arm are a diagnostic noise check, not a variance estimate.",
            "Manual fact checks are needed; archived strict judge labels are not reused as POC results.",
        ],
    }
    for case_id, spec in SWAPS.items():
        row = next(item for item in packet["cases"] if item["case_id"] == case_id)
        base = [dict(memory) for memory in row["base_context"]]
        candidate = next(memory for memory in row["graph_context"] if memory["id"] == spec["graph_memory_id"])
        swap = [dict(memory) for memory in base]
        removed = swap[spec["replace_rank"] - 1]
        swap[spec["replace_rank"] - 1] = dict(candidate, rank=spec["replace_rank"])
        assert len(base) == len(swap) == 15
        assert sum(a["content"] != b["content"] for a, b in zip(base, swap)) == 1
        case = {
            "case_id": case_id, "question": row["question"], "gold": row["gold"],
            "replace_rank": spec["replace_rank"],
            "removed_memory": {"id": removed["id"], "content": removed["content"]},
            "added_memory": {"id": candidate["id"], "content": candidate["content"]},
            "base_context_ids": [memory["id"] for memory in base],
            "swap_context_ids": [memory["id"] for memory in swap],
            "calls": [],
        }
        result["cases"].append(case)
        prompts = {arm: prompt(row["question"], context) for arm, context in [("base", base), ("swap", swap)]}
        case["prompt_sha256"] = {arm: hashlib.sha256(value.encode()).hexdigest() for arm, value in prompts.items()}
        for call_index, arm in enumerate(ORDER, 1):
            try:
                answer, usage = client.complete(
                    system=SYSTEM, user=prompts[arm], max_tokens=1200, temperature=0.0
                )
                case["calls"].append({"index": call_index, "arm": arm, "answer": answer, "usage": usage})
                print(case_id, call_index, arm, answer[:120].replace("\n", " "), flush=True)
            except Exception as exc:
                case["calls"].append({"index": call_index, "arm": arm, "error_type": type(exc).__name__, "error": str(exc)[:200]})
                OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                raise
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
