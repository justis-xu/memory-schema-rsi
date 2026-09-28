"""Audit one fixed Chinese time question against source, contexts and judge votes.

Answer classifications are manual source judgments, not a new judge run.
"""

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZH = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
EN = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10.json")
OUTPUT = ROOT / "results/analysis/m2_trip_date_judge_20260928.json"
CASE_ID = "locomo_conv-26_qa74"
EARLY_ID = "7da557c9-eb9c-4392-b9c5-2ac9863d2695"
LATE_ID = "f87df58f-81e4-46e4-a2bc-22e62b98c625"
HIKE_ID = "65dc20b5-1cc1-4c95-ac07-40f204046256"

# These labels classify only the response's date wording against D18:1.
# They do not judge all other facts or attribute the answer to one memory.
LABELS = {
    "zhfull_fused_jevdec_locomo_20260925_202122.jsonl": ("unsupported_weekday_precision", "把上周末约成10月18–19日，和周末日历不一致"),
    "zhfull_fused_jevlaya_locomo_20260925_205035.jsonl": ("explicit_report_day_as_trip_day", "把10月20日写成出游日"),
    "zhfull_fused_locomo_20260925_161049.jsonl": ("source_compatible_core", "保留上周末；10月19日前不久虽宽泛但未明确改成当天"),
    "zhfull_fused_locomo_20260925_172126.jsonl": ("source_compatible_core", "保留10月20日前的上周末，未给出冲突的具体日"),
    "zhfull_fused_locomo_20260925_195248.jsonl": ("source_compatible_core", "保留10月20日前的上周末，未给出冲突的具体日"),
    "zhfull_locomo_20260925_115806.jsonl": ("unsupported_weekday_precision", "虽写上周末，又约成10月19日前后"),
    "zhfull_locomo_20260925_151059.jsonl": ("unsupported_weekday_precision", "把旅行推成10月18–20日期间，并以19日徒步倒推"),
    "zhfull_locomo_20260925_164645.jsonl": ("explicit_report_day_as_trip_day", "直接答10月20日并称旅行本身在该日前后"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_case(path):
    case, = [item for item in json.loads(path.read_text()) if item["sample_id"] == "conv-26"]
    session = case["conversation"]["session_18"]
    turn, = [item for item in session if item["dia_id"] == "D18:1"]
    hike_turns = [item for item in session if item["dia_id"] in {"D18:15", "D18:16", "D18:17"}]
    assert len(hike_turns) == 3
    qa = case["qa"][74]
    return {"session_date": case["conversation"]["session_18_date_time"],
            "turn": turn, "hike_turns": hike_turns, "qa": qa}


def main():
    assert date(2023, 10, 20).weekday() == 4  # Friday
    source = {"zh": source_case(ZH), "en": source_case(EN)}
    assert source["zh"]["qa"]["evidence"] == source["en"]["qa"]["evidence"] == ["D18:1"]
    assert "上周末" in source["zh"]["turn"]["text"]
    assert "this past weekend" in source["en"]["turn"]["text"]
    assert source["zh"]["qa"]["answer"] == "2023年10月20日之前的那个周末"
    rows = []
    canonical_context = None
    canonical_hash = None
    for path in sorted((ROOT / "results_zh").glob("zhfull*jsonl")):
        if path.name not in LABELS:
            continue
        matches = []
        for line in path.open():
            row = json.loads(line)
            if row.get("case_id") == CASE_ID:
                matches.append(row)
        assert len(matches) == 1, (path, len(matches))
        row = matches[0]
        by_id = {memory["id"]: memory for memory in row["retrieved_memories"] + row["graph_memories"]}
        ids = row["metadata"]["answer_context_ids"]
        assert len(ids) == 15 and len(set(ids)) == 15
        context = [{"id": mid, "content": by_id[mid]["content"], "metadata": by_id[mid]["metadata"]} for mid in ids]
        context_hash = digest(context)
        if canonical_context is None:
            canonical_context, canonical_hash = context, context_hash
        else:
            assert context_hash == canonical_hash, (path.name, context_hash, canonical_hash)
        assert EARLY_ID in ids and LATE_ID in ids and HIKE_ID in ids
        assert ids.index(EARLY_ID) == 0 and ids.index(LATE_ID) == 1 and ids.index(HIKE_ID) == 3
        assert row["expected_answer"] == source["zh"]["qa"]["answer"]
        assert row["metrics"]["judge_votes"] == [True] and row["metrics"]["judge_correct"] is True
        assert row["metrics"]["exact"] is False
        label, reason = LABELS[path.name]
        rows.append({"run_file": str(path.relative_to(ROOT)), "case_record_sha256": digest(row),
                     "graph_enabled": row["metadata"].get("graph_enabled"),
                     "graph_context_ids": row["metadata"].get("graph_context_ids"),
                     "context_sha256": context_hash, "answer_context_ids": ids,
                     "predicted_answer": row["predicted_answer"], "metrics": row["metrics"],
                     "manual_date_label": label, "manual_reason_zh": reason})
    assert len(rows) == len(LABELS) == 8
    assert all(not row["graph_context_ids"] for row in rows)
    categories = {label: sum(row["manual_date_label"] == label for row in rows) for label in sorted({row["manual_date_label"] for row in rows})}
    assert categories == {"explicit_report_day_as_trip_day": 2, "source_compatible_core": 3,
                          "unsupported_weekday_precision": 3}
    payload = {"scope": "Fixed conv-26_qa74, eight archived Chinese runs with identical visible 15-memory final context; manual date wording audit, no model calls",
               "sources": source, "source_sha256": {"zh": hashlib.sha256(ZH.read_bytes()).hexdigest(),
                                                     "en": hashlib.sha256(EN.read_bytes()).hexdigest()},
               "calendar": {"report_day": "2023-10-20", "weekday": "Friday",
                            "ordinary_previous_weekend": ["2023-10-14", "2023-10-15"]},
               "common_context_sha256": canonical_hash, "common_context": canonical_context,
               "key_context_positions": {"source_session_memory": {"id": EARLY_ID, "zero_based": 0},
                                         "later_misdated_memory": {"id": LATE_ID, "zero_based": 1},
                                         "hike_next_day_relation_memory": {"id": HIKE_ID, "zero_based": 3,
                                             "manual_source_note_zh": "D18:17只说徒步/照片昨天、是公路旅行后放松；没有说公路旅行后的次日"}},
               "counts": {"runs": len(rows), "all_judge_true": sum(row["metrics"]["judge_correct"] for row in rows),
                          "manual_date_labels": categories}, "runs": rows,
               "limits": ["Same visible context ID/content/metadata does not prove byte-identical historical prompts, model state, or judge state",
                          "Wrong or unsupported date wording in an answer does not prove the later memory alone caused it; the earlier correct memory was also present",
                          "Single question and eight correlated runs do not estimate judge-wide false-positive rate"]}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
