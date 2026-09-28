#!/usr/bin/env python3
"""Offline structural write gate for archived source-grounded merge outputs."""

import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/analysis/m2_carryover_live_merge_v1_20260928.json"
REVIEW = ROOT / "results/analysis/m2_carryover_live_merge_review_20260928.json"
OUTPUT = ROOT / "results/analysis/m2_merge_gate_offline_20260928.json"
FENCE = re.compile(r"```json\s*(\{.*\})\s*```", re.S)
SOURCE_ID = re.compile(r"\[(D\d+:\d+) (?:text|image_caption|image_query)\]")
REQUIRED = {"case_id", "claim_zh", "support", "source_turn_ids", "source_session_id",
            "source_speakers", "covered_by_memory_ids", "action"}
SUPPORT = {"full", "partial", "absent"}
ACTIONS = {"保留", "合并", "撤回"}


def add(reasons: list[dict], code: str, fact_index: int | None = None) -> None:
    reason = {"code": code}
    if fact_index is not None:
        reason["fact_index"] = fact_index
    reasons.append(reason)


def inspect_output(request: dict, call: dict, fixture: dict) -> dict:
    reasons: list[dict] = []
    raw = call.get("raw_response", "")
    if call.get("json_parse_ok"):
        parsed = call["parsed_response"]
        parse_status = "strict_json"
    else:
        add(reasons, "invalid_strict_json")
        match = FENCE.fullmatch(raw)
        parsed = json.loads(match.group(1)) if match else None
        parse_status = "fenced_json_diagnostic_only" if match else "unparseable"
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        add(reasons, "invalid_facts_container")
        return {"parse_status": parse_status, "verdict": "reject", "reasons": reasons}

    source_ids = set(SOURCE_ID.findall(request["user"]))
    memory_cases = {item["memory_id"]: item["case_id"] for item in fixture["candidates"]}
    case_ids = set(fixture["case_ids"])
    for index, fact in enumerate(parsed["facts"]):
        if not isinstance(fact, dict) or REQUIRED - set(fact):
            add(reasons, "missing_required_fields", index)
            continue
        if fact["case_id"] not in case_ids:
            add(reasons, "unknown_case_id", index)
        if not isinstance(fact["claim_zh"], str) or not fact["claim_zh"].strip():
            add(reasons, "empty_claim", index)
        if fact["support"] not in SUPPORT or fact["action"] not in ACTIONS:
            add(reasons, "invalid_support_or_action", index)
            continue
        if not isinstance(fact["source_turn_ids"], list) or not fact["source_turn_ids"]:
            add(reasons, "missing_source_turn", index)
            continue
        for source_id in fact["source_turn_ids"]:
            if source_id not in source_ids:
                add(reasons, "source_turn_outside_input", index)
            if not isinstance(source_id, str) or not re.fullmatch(r"D\d+:\d+", source_id):
                add(reasons, "bad_source_turn_shape", index)
                continue
            if fact["source_session_id"] != "session_" + source_id.split(":")[0][1:]:
                add(reasons, "source_session_mismatch", index)
        covered = fact["covered_by_memory_ids"]
        if not isinstance(covered, list):
            add(reasons, "bad_coverage_shape", index)
            continue
        for memory_id in covered:
            if memory_id not in memory_cases:
                add(reasons, "memory_id_outside_candidates", index)
            elif memory_cases[memory_id] != fact["case_id"]:
                add(reasons, "cross_case_memory_id", index)
        if fact["support"] == "absent" and fact["action"] != "撤回":
            add(reasons, "absent_fact_accepted", index)
        if fact["support"] == "partial" and fact["action"] != "撤回":
            add(reasons, "partial_fact_accepted_without_review", index)
        if fact["support"] == "full" and not covered and fact["action"] == "撤回":
            add(reasons, "source_only_fact_called_retraction", index)
    return {"parse_status": parse_status, "fact_count": len(parsed["facts"]),
            "verdict": "reject" if reasons else "structural_pass",
            "reasons": reasons}


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing overwrite: {OUTPUT}")
    run_bytes = RUN.read_bytes()
    review = json.loads(REVIEW.read_text())
    run = json.loads(run_bytes)
    fixtures = {item["fixture_id"]: item for item in run["fixtures"]}
    outcomes = []
    for request, call in zip(run["requests"], run["calls"]):
        result = inspect_output(request, call, fixtures[call["fixture_id"]])
        outcomes.append({"call_index": call["call_index"], "fixture_id": call["fixture_id"],
                         "arm": call["arm"], **result})
    counts = {"calls": len(outcomes), "structural_pass": sum(x["verdict"] == "structural_pass" for x in outcomes),
              "rejected": sum(x["verdict"] == "reject" for x in outcomes)}
    assert counts == {"calls": 6, "structural_pass": 1, "rejected": 5}, counts
    assert outcomes[3]["verdict"] == "structural_pass" and outcomes[3]["fixture_id"] == "melanie_pottery"
    assert review["case_reviews"][2]["base"]["qualification_but_ok_retained"] is False
    # Sensitivity check only: remove the unrelated partial date row from the
    # road base output. The gate then passes although three false friend claims
    # remain. This mutated output is not a model response or a runtime result.
    road_request = run["requests"][4]
    road_call = copy.deepcopy(run["calls"][4])
    dropped = [fact for fact in road_call["parsed_response"]["facts"] if fact["support"] == "partial"]
    assert len(dropped) == 1 and "2022" in dropped[0]["claim_zh"]
    road_call["parsed_response"]["facts"] = [
        fact for fact in road_call["parsed_response"]["facts"] if fact is not dropped[0]
    ]
    friend_claims_remaining = sum("朋友" in fact["claim_zh"] for fact in road_call["parsed_response"]["facts"])
    assert friend_claims_remaining == 3
    sensitivity = inspect_output(road_request, road_call, fixtures["road_companion_current"])
    assert sensitivity["verdict"] == "structural_pass"
    report = {"scope": "offline structural write gate on six archived outputs; no model, DB, graph or answer calls",
              "run_sha256": hashlib.sha256(run_bytes).hexdigest(), "counts": counts, "outcomes": outcomes,
              "manual_comparison": {
                  "structural_pass_call_index": 4,
                  "known_semantic_omission": "陶艺基础臂遗漏本人同轮“但我还好”限定语；结构门槛仍放行",
                  "not_a_runtime_oracle": True,
              },
              "counterfactual_sensitivity_not_model_output": {
                  "mutation": "remove only the unrelated partial 2022 date fact from road base output",
                  "friend_claims_remaining": friend_claims_remaining,
                  "gate_verdict_after_mutation": sensitivity["verdict"],
                  "meaning": "A structural pass would not prove the accepted friend identity is supported.",
              },
              "limits": ["The gate verifies syntax and claimed IDs/actions only, not semantic entailment or memory-text coverage.",
                         "Rejecting partial results is a conservative offline policy, not a measured production threshold.",
                         "All source sessions were supplied in the prior model POC; retrieval and QA benefit remain untested."]}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
