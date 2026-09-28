#!/usr/bin/env python3
"""Recheck fixed live-merge output structure without another model call."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/analysis/m2_carryover_live_merge_v1_20260928.json"
REVIEW = ROOT / "results/analysis/m2_carryover_live_merge_review_20260928.json"
FENCE = re.compile(r"```json\s*(\{.*\})\s*```", re.S)
SOURCE_ID = re.compile(r"\[(D\d+:\d+) (?:text|image_caption|image_query)\]")


def main() -> None:
    run = json.loads(RUN.read_text())
    review = json.loads(REVIEW.read_text())
    assert len(run["requests"]) == len(run["calls"]) == run["max_calls"] == 6
    assert run["completed_planned_calls"]
    fixtures = {item["fixture_id"]: item for item in run["fixtures"]}
    counts = {"strict_json_responses": 0, "fully_fenced_json_responses": 0,
              "output_facts_after_fence_salvage": 0, "source_refs_not_in_supplied_turns": 0,
              "source_session_mismatches": 0, "cross_case_memory_id_claims": 0,
              "prompt_tokens": 0, "completion_tokens": 0}
    for request, call in zip(run["requests"], run["calls"]):
        assert request["fixture_id"] == call["fixture_id"] and request["arm"] == call["arm"]
        assert "manual_acceptance" not in request["user"] and "in_older_memory" not in request["user"]
        counts["prompt_tokens"] += call["usage"]["prompt_tokens"]
        counts["completion_tokens"] += call["usage"]["completion_tokens"]
        if call["json_parse_ok"]:
            counts["strict_json_responses"] += 1
            parsed = call["parsed_response"]
        else:
            match = FENCE.fullmatch(call["raw_response"])
            assert match, call["call_index"]
            counts["fully_fenced_json_responses"] += 1
            parsed = json.loads(match.group(1))
        allowed_sources = set(SOURCE_ID.findall(request["user"]))
        memory_cases = {item["memory_id"]: item["case_id"] for item in fixtures[call["fixture_id"]]["candidates"]}
        facts = parsed["facts"]
        counts["output_facts_after_fence_salvage"] += len(facts)
        for fact in facts:
            for source_id in fact.get("source_turn_ids", []):
                counts["source_refs_not_in_supplied_turns"] += source_id not in allowed_sources
                counts["source_session_mismatches"] += (
                    fact.get("source_session_id") != "session_" + source_id.split(":")[0][1:]
                )
            for memory_id in fact.get("covered_by_memory_ids", []):
                assert memory_id in memory_cases
                counts["cross_case_memory_id_claims"] += memory_cases[memory_id] != fact["case_id"]
    assert all(counts[key] == review[key] for key in counts), counts
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
