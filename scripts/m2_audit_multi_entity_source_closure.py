#!/usr/bin/env python3
"""Build a source/evidence packet for the fixed eight Chinese two-person questions."""

import hashlib
import json
from pathlib import Path

from m2_audit_zh_graph_cache_alignment import current_memories, signature


ROOT = Path(__file__).resolve().parents[1]
ZH = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
EN = ROOT / "data/locomo/locomo10.json"
COHORT = ROOT / "results/analysis/m2_multi_entity_cohort_retrieval_20260927.json"
OUT = ROOT / "results/analysis/m2_multi_entity_source_closure_20260927.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(sample, evidence):
    out = []
    for sid, rows in sample["conversation"].items():
        if not sid.startswith("session_") or sid.endswith("_date_time"):
            continue
        for i, row in enumerate(rows):
            if row["dia_id"] not in evidence:
                continue
            out.append({"dia_id": row["dia_id"], "session_id": sid,
                        "speaker": row["speaker"], "text": row.get("text"),
                        "blip_caption": row.get("blip_caption"),
                        "previous_turn": {"dia_id": rows[i-1]["dia_id"],
                                          "speaker": rows[i-1]["speaker"],
                                          "text": rows[i-1].get("text")}
                                         if i > 0 else None,
                        "next_turn": {"dia_id": rows[i+1]["dia_id"],
                                      "speaker": rows[i+1]["speaker"],
                                      "text": rows[i+1].get("text")}
                                     if i + 1 < len(rows) else None})
    if {r["dia_id"] for r in out} != set(evidence):
        raise ValueError((sample["sample_id"], evidence))
    return sorted(out, key=lambda x: (int(x["session_id"].split("_")[1]),
                                      int(x["dia_id"].split(":")[1])))


def title_mentions(sample, needle):
    return [{"dia_id": row["dia_id"], "speaker": row["speaker"],
             "text": row.get("text")}
            for sid, rows in sample["conversation"].items()
            if sid.startswith("session_") and not sid.endswith("_date_time")
            for row in rows if needle.lower() in (row.get("text") or "").lower()]


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    zh = {s["sample_id"]: s for s in json.loads(ZH.read_text())}
    en = {s["sample_id"]: s for s in json.loads(EN.read_text())}
    cohort = json.loads(COHORT.read_text())
    memories = current_memories()
    cases = []
    for c in cohort["cases"]:
        conv = c["conversation_id"]
        qi = int(c["case_id"].split("_qa")[-1])
        qz, qe = zh[conv]["qa"][qi], en[conv]["qa"][qi]
        assert c["question"] == qz["question"]
        assert qz["evidence"] == qe["evidence"]
        evidence_sessions = {"session_" + dia.split(":")[0][1:]
                             for dia in qz["evidence"]}
        source_memories = [
            {"id": mid, "session_id": row["session_id"], "content": row["content"]}
            for mid, row in memories.items()
            if row["user_id"] == c["user_id"] and row["session_id"] in evidence_sessions
        ]
        movie_mentions = ({"zh_lord_of_the_rings": title_mentions(zh[conv], "指环王"),
                           "en_lord_of_the_rings": title_mentions(en[conv], "Lord of the Rings")}
                          if c["case_id"] == "locomo_conv-42_qa42" else None)
        cases.append({"case_id": c["case_id"], "user_id": c["user_id"],
                      "zh_question": qz["question"], "zh_answer": qz["answer"],
                      "en_question": qe["question"], "en_answer": qe["answer"],
                      "evidence_ids": qz["evidence"],
                      "zh_evidence_with_neighbors": source_rows(zh[conv], qz["evidence"]),
                      "en_evidence_with_neighbors": source_rows(en[conv], qe["evidence"]),
                      "current_evidence_session_memories": source_memories,
                      "whole_conversation_title_mentions": movie_mentions,
                      "original_top15_ids": [r["id"] for r in c["arms"]["original"]["reranked"]]})
    assert len(cases) == 8
    report = {"scope": "All eight previously question-text-selected Chinese two-person category-1 cases; bilingual gold/evidence, adjacent source turns and current memories in evidence sessions; no model calls",
              "input_sha256": {"zh": sha(ZH), "en": sha(EN), "cohort": sha(COHORT)},
              "current_memory_signature": signature(memories),
              "cases": cases,
              "limits": ["Evidence IDs can omit background turns necessary to interpret a reference or status.",
                         "Current memory records lack exact dia_id provenance; evidence-session grouping does not prove turn-to-memory derivation.",
                         "Only the fixed eight question-text-selected cases are audited; no population rate is estimated."]}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("cases", len(cases), "evidence turns",
          sum(len(c["zh_evidence_with_neighbors"]) for c in cases))


if __name__ == "__main__":
    main()
