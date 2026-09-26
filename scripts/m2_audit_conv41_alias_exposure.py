"""Count observed User/John/约翰 graph-candidate exposure in one Chinese fused run."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results_zh/zhfull_fused_locomo_20260925_161049.jsonl"
ALIASES = ROOT / "results/analysis/m2_conv41_alias_hub_20260926.json"
OUT = ROOT / "results/analysis/m2_conv41_alias_exposure_20260926.json"


def main():
    user_ids = {x["memory_id"] for x in json.loads(ALIASES.read_text())["user_memories"]}
    counts = Counter()
    tags = ("entity:User", "entity:John", "entity:约翰")
    distinct = defaultdict(set)
    cases = []
    for line in RUN.open():
        row = json.loads(line)
        if not row["case_id"].startswith("locomo_conv-41_"):
            continue
        counts["questions"] += 1
        retrieved = row.get("retrieved_memories") or []
        graph = row.get("graph_memories") or []
        context = set(row.get("metadata", {}).get("answer_context_ids") or [])
        counts["user_memory_final_context_occurrences"] += sum(mid in user_ids for mid in context)
        counts["user_memory_final_context_questions"] += bool(context & user_ids)
        counts["user_memory_retrieved_occurrences"] += sum(m["id"] in user_ids for m in retrieved)
        anchored = [m for m in graph if m.get("anchor_memory_id") in user_ids]
        counts["graph_candidate_occurrences_anchored_by_user_memory"] += len(anchored)
        counts["questions_with_user_memory_anchor"] += bool(anchored)
        case = {"case_id": row["case_id"], "category": row.get("metadata", {}).get("category"),
                "question": row["question"], "gold": row.get("expected_answer"),
                "prediction": row.get("predicted_answer"), "alias_candidates": []}
        for tag in tags:
            hits = [m for m in graph if tag in m.get("via", [])]
            counts[f"{tag}_candidate_occurrences"] += len(hits)
            counts[f"{tag}_candidate_questions"] += bool(hits)
            counts[f"{tag}_selected_occurrences"] += sum(m["id"] in context for m in hits)
            counts[f"{tag}_selected_questions"] += any(m["id"] in context for m in hits)
            distinct[tag].update(m["id"] for m in hits)
            if tag == "entity:User":
                case["alias_candidates"] = [
                    {"id": m["id"], "content": m["content"], "selected": m["id"] in context,
                     "anchor_memory_id": m.get("anchor_memory_id"), "via": m.get("via")}
                    for m in hits
                ]
        if case["alias_candidates"]:
            cases.append(case)
    for tag in tags:
        counts[f"{tag}_distinct_candidate_ids"] = len(distinct[tag])
    assert counts["questions"] == 192
    assert len(cases) == counts["entity:User_candidate_questions"]
    OUT.write_text(json.dumps({
        "scope": "One archived Chinese fused Graph run, conv-41 only; observed candidate tags and selected IDs",
        "run": str(RUN.relative_to(ROOT)), "counts": dict(counts), "cases_with_user_entity_candidate": cases,
        "limits": ["The archive does not contain the full pre-graph seed list or untruncated node neighbors",
                   "A via tag is one retrieval route, not proof that the candidate uniquely answered the question",
                   "This is one historical graph build, not a controlled alias-merge replay"],
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
