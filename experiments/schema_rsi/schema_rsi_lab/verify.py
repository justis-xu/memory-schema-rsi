"""RSIAgent-style local verify/repair loop over a frozen graph policy.

The verifier sees no gold answer. It distinguishes an unsupported answer from
one contradicted by evidence already in context. Only unsupported answers
trigger graph traversal; a candidate is committed only after re-verification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.memory.base import MemoryRecord

from .lifecycle import LifecycleGraph
from .provenance import SourceGraph
from .success_paths import SuccessPathGraph, budgeted_route


_JSON = re.compile(r"\{.*\}", re.S)


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str
    search_query: str
    model_called: bool = True
    support_ids: tuple[str, ...] = ()


def verify_answer(client, question: str, answer: str, memories: list[MemoryRecord]) -> Verdict:
    if re.search(r"\b(?:not|isn't|aren't) (?:recorded|detailed|specified|known|provided)|\b(?:don't|doesn't) (?:record|know|say)|\bno (?:evidence|indication)\b", answer, re.I):
        return Verdict("missing", "answer explicitly says the requested detail is unavailable", question, False)
    context = "\n".join(f"[{i}] {m.content}" for i, m in enumerate(memories, 1))
    prompt = (
        "Check whether the proposed answer is fully supported by the listed memories. "
        "Do not use outside knowledge. Judge the actual question, including all requested "
        "items and dates. A memory that merely mentions the topic is not evidence.\n"
        "If the proposed answer says the requested detail is unknown or absent, "
        "classify it as missing when the question asks for that detail.\n"
        "Return only JSON with keys status, reason, search_query, support_indices. "
        "support_indices must list the numbered memories that together support "
        "the whole answer when status is supported; otherwise use an empty list. "
        "status must be one of "
        "supported, contradicted, missing, uncertain. Use contradicted when the listed "
        "memories support a different or more complete answer. Use missing when the needed "
        "evidence is absent. search_query should be a brief query for missing evidence, "
        "or empty if no evidence is missing.\n\n"
        f"Question: {question}\nProposed answer: {answer}\nMemories:\n{context}"
    )
    text, _ = client.complete(system="You are a strict evidence verifier.", user=prompt, max_tokens=400)
    match = _JSON.search(text)
    if not match:
        return Verdict("uncertain", "verifier did not return JSON", "")
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return Verdict("uncertain", "invalid verifier JSON", "")
    status = str(obj.get("status", "uncertain")).lower()
    if status not in {"supported", "contradicted", "missing", "uncertain"}:
        status = "uncertain"
    support_ids = []
    if status == "supported" and isinstance(obj.get("support_indices"), list):
        for raw_index in obj["support_indices"]:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(memories) and memories[index - 1].id not in support_ids:
                support_ids.append(memories[index - 1].id)
    return Verdict(status, str(obj.get("reason", ""))[:500],
                   str(obj.get("search_query", ""))[:200], True, tuple(support_ids))


def repair_case(case: dict, memories: dict[str, dict], graph: LifecycleGraph,
                sources: SourceGraph,
                answerer: Answerer, verifier_client, *, reference_date: str | None,
                success_paths: SuccessPathGraph | None = None) -> dict:
    user = memories[case["base_ranked"][0]]["user_id"]
    def records(ids: list[str]) -> list[MemoryRecord]:
        return [MemoryRecord(id=mid, content=(
                    f"Source conversation on {sources.turns[mid[2:]]['session_date']}: "
                    f"{sources.turns[mid[2:]]['content']}" if mid.startswith("T:") else memories[mid]["content"]),
                    metadata={} if mid.startswith("T:") else memories[mid].get("metadata", {}))
                for mid in ids]

    baseline_ids = case["base_ranked"][:15]
    baseline_context = records(baseline_ids)
    if not graph.policy.verify_enabled:
        return {"id": case["id"], "category": case["category"],
                "baseline_answer": case["baseline_answer"], "baseline_correct": case["baseline_correct"],
                "action": "keep", "selected": baseline_ids, "graph_added": [],
                "final_answer": case["baseline_answer"], "accepted": False}
    first = verify_answer(verifier_client, case["question"], case["baseline_answer"], baseline_context)
    result = {"id": case["id"], "category": case["category"], "baseline_answer": case["baseline_answer"],
              "baseline_correct": case["baseline_correct"], "first_verdict": first.__dict__,
              "action": "keep", "selected": baseline_ids, "graph_added": [],
              "final_answer": case["baseline_answer"], "accepted": False,
              "model_calls": int(first.model_called), "edge_visits": 0}
    if first.status == "supported":
        result["support_ids"] = list(first.support_ids)
    if first.status in {"supported", "uncertain"}:
        return result

    selected = baseline_ids
    if first.status == "missing":
        query = first.search_query or case["question"]
        retrieval = graph.for_question(user, query, baseline_ids, context_k=15)
        provenance = (sources.for_question(user, case["question"], baseline_ids)
                      if graph.policy.use_source else {"source_turns": []})
        turn_ids = [turn["id"] for turn in provenance["source_turns"]]
        selected = retrieval["selected"]
        selected = selected[:max(0, 15 - len(turn_ids))] + turn_ids
        if graph.policy.use_success_paths and success_paths is not None:
            path = success_paths.query(user=user, question=case["question"],
                                       threshold=graph.policy.success_path_threshold,
                                       exclude_episode=case["id"])
            augmented = budgeted_route(selected, path["support_ids"],
                                       slots=graph.policy.success_path_slots)
            result["success_path_peers"] = path["peer_ids"]
            result["success_path_added"] = [mid for mid in augmented if mid not in selected]
            selected = augmented
        if not retrieval["graph_added"] and not turn_ids and not result.get("success_path_added"):
            result["action"] = "no_graph_or_source_candidate"
            return result
        result["graph_added"] = retrieval["graph_added"]
        result["edge_visits"] = retrieval["edge_visits"]
        result["source_added"] = provenance["source_turns"]
        result["action"] = ("success_path_reanswer" if result.get("success_path_added")
                            else "source_graph_reanswer" if turn_ids else "graph_reanswer")
    else:
        result["action"] = "same_context_reanswer"

    result["selected"] = selected
    proposal, _ = answerer.answer(case["question"], records(selected), reference_date=reference_date,
                                  benchmark="locomo")
    result["model_calls"] += 1
    result["proposed_answer"] = proposal
    second = verify_answer(verifier_client, case["question"], proposal, records(selected))
    result["model_calls"] += int(second.model_called)
    result["second_verdict"] = second.__dict__
    if second.status == "supported":
        result["accepted"] = True
        result["final_answer"] = proposal
        result["support_ids"] = list(second.support_ids)
    return result
