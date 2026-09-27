#!/usr/bin/env python3
"""Screen reconstructible Chinese Jev states for answer literals hidden after per-memory char 60."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZH = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
ARMS = {
    "decider": ROOT / "results_zh/zhfull_fused_jevdec_locomo_20260925_202122.jsonl",
    "laya": ROOT / "results_zh/zhfull_fused_jevlaya_locomo_20260925_205035.jsonl",
}
OUT = ROOT / "results/analysis/m2_jev_hidden_answer_tails_20260927.json"
AUDIT_CASES = {"locomo_conv-42_qa102", "locomo_conv-44_qa114",
               "locomo_conv-47_qa122", "locomo_conv-48_qa26"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_turns(sample, evidence):
    turns = []
    for sid, rows in sample["conversation"].items():
        if not sid.startswith("session_") or sid.endswith("_date_time"):
            continue
        for row in rows:
            if row["dia_id"] in evidence:
                turns.append({"dia_id": row["dia_id"], "session_id": sid,
                              "speaker": row["speaker"], "text": row.get("text"),
                              "blip_caption": row.get("blip_caption")})
    assert {t["dia_id"] for t in turns} == set(evidence)
    return turns


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    source = {x["sample_id"]: x for x in json.loads(ZH.read_text())}
    result = {
        "scope": "Literal Chinese gold substring screen in exact-reconstructible non-expanded historical Jev states; four source-audited cases selected post screen; no model calls",
        "screen_rule": "2<=stripped gold chars<=15; gold appears in at least one full final-context memory, no first-60 memory prefix contains the full literal; first score numeric and no expansion",
        "input_sha256": {"dataset": sha(ZH), **{arm: sha(path) for arm, path in ARMS.items()}},
        "arms": {},
        "limits": ["Literal nonvisibility does not prove all paraphrases or implied facts are absent from the digest.",
                   "Screen excludes expanded cases because first ordered context was not archived, and excludes long/multi-part gold strings.",
                   "Strict exact scores and model answers are archived observations, not effects of changing Jev input.",
                   "The four detailed cases were selected after seeing the screen; no population prevalence follows."],
    }
    for arm, path in ARMS.items():
        eligible = 0
        matched = []
        detail = []
        for line in path.open():
            row = json.loads(line)
            jev = (row.get("metadata") or {}).get("jev_stop") or {}
            if not isinstance(jev.get("first"), (float, int)) or jev.get("expanded"):
                continue
            eligible += 1
            gold = row["expected_answer"].strip().strip('。.!！"《》 ')
            if not (2 <= len(gold) <= 15):
                continue
            contents = {m["id"]: m["content"] for m in
                        row["retrieved_memories"] + row.get("graph_memories", [])}
            ids = row["metadata"]["answer_context_ids"][:15]
            if any(mid not in contents for mid in ids):
                continue
            if not any(gold in contents[mid] for mid in ids):
                continue
            if any(gold in contents[mid][:60] for mid in ids):
                continue
            matched.append({"case_id": row["case_id"], "question": row["question"],
                            "gold_literal": gold, "first_score": jev["first"],
                            "strict_exact": row["metrics"].get("exact")})
            if row["case_id"] not in AUDIT_CASES:
                continue
            conv = row["case_id"].split("_qa")[0].replace("locomo_", "")
            qi = int(row["case_id"].split("_qa")[-1])
            qa = source[conv]["qa"][qi]
            assert qa["question"] == row["question"]
            digest = " | ".join(contents[mid][:60] for mid in ids)
            detail.append({"case_id": row["case_id"], "question": row["question"],
                           "gold": row["expected_answer"],
                           "predicted_answer": row["predicted_answer"],
                           "first_score": jev["first"], "expanded": jev["expanded"],
                           "strict_exact": row["metrics"].get("exact"),
                           "source_evidence": source_turns(source[conv], qa["evidence"]),
                           "digest": digest, "digest_sha256": hashlib.sha256(digest.encode()).hexdigest(),
                           "context": [{"rank": rank, "id": mid, "prefix": contents[mid][:60],
                                        "tail": contents[mid][60:],
                                        "gold_literal_start": contents[mid].find(gold)}
                                       for rank, mid in enumerate(ids, 1)]})
        result["arms"][arm] = {"path": str(path.relative_to(ROOT)),
                                "scored_unexpanded_count": eligible,
                                "literal_screen_match_count": len(matched),
                                "literal_screen_matches": matched,
                                "source_audit_cases": detail}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for arm, row in result["arms"].items():
        print(arm, row["literal_screen_match_count"], "/", row["scored_unexpanded_count"],
              "detailed", len(row["source_audit_cases"]))


if __name__ == "__main__":
    main()
