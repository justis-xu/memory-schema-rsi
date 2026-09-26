"""Compare locally downloaded English/Chinese benchmark structure without model calls."""

import gc
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/xu/git/eval-datasets")
OUT = ROOT / "results/analysis/m2_hf_dataset_coverage_20260926.json"


def locomo_signature(path: Path):
    raw = json.loads(path.read_text())
    signatures = {}
    counts = Counter()
    for c in raw:
        sid = str(c["sample_id"])
        conv = c["conversation"]
        sessions = {k: v for k, v in conv.items() if k.startswith("session_") and not k.endswith("_date_time")}
        signatures[sid] = {
            "sessions": {k: {"date": conv.get(k + "_date_time"), "dia_ids": [t.get("dia_id") for t in turns], "speakers": [t.get("speaker") for t in turns]} for k, turns in sessions.items()},
            "qa": [{"category": q.get("category"), "evidence": q.get("evidence"), "has_answer": "answer" in q} for q in c["qa"]],
        }
        counts["sessions"] += len(sessions)
        counts["turns"] += sum(len(v) for v in sessions.values())
        counts["questions"] += len(c["qa"])
        for q in c["qa"]:
            counts[f"category_{q['category']}"] += 1
    counts["conversations"] = len(raw)
    del raw
    gc.collect()
    return signatures, dict(counts)


def lme_signature(path: Path):
    raw = json.loads(path.read_text())
    signatures = {}
    counts = Counter()
    for q in raw:
        qid = str(q["question_id"])
        sids = q.get("haystack_session_ids") or []
        dates = q.get("haystack_dates") or []
        sessions = q.get("haystack_sessions") or []
        if not (len(sids) == len(dates) == len(sessions)):
            counts["malformed_parallel_lists"] += 1
        signatures[qid] = {
            "question_type": q.get("question_type"),
            "question_date": q.get("question_date"),
            "answer_session_ids": q.get("answer_session_ids") or [],
            "sessions": {str(sid): {"date": dates[i], "roles": [m.get("role") for m in sessions[i]]} for i, sid in enumerate(sids)},
        }
        counts["sessions"] += len(sids)
        counts["messages"] += sum(len(s) for s in sessions)
        counts[f"type_{q.get('question_type')}"] += 1
        counts["answer_sessions_missing"] += len(set(q.get("answer_session_ids") or []) - set(sids))
    counts["questions"] = len(raw)
    del raw
    gc.collect()
    return signatures, dict(counts)


def main():
    le, lec = locomo_signature(ROOT / "data/locomo/locomo10.json")
    lz, lzc = locomo_signature(DATA / "locomo-zh/locomo10_zh.json")
    locomo = {
        "english": lec, "chinese": lzc,
        "sample_ids_equal": set(le) == set(lz),
        "session_ids_dates_dia_ids_equal": all(
            {k: (v["date"], v["dia_ids"]) for k, v in le[s]["sessions"].items()}
            == {k: (v["date"], v["dia_ids"]) for k, v in lz[s]["sessions"].items()}
            for s in set(le) & set(lz)),
        "qa_categories_evidence_answer_presence_equal": all(le[s]["qa"] == lz[s]["qa"] for s in set(le) & set(lz)),
    }
    del le, lz
    gc.collect()

    ee, eec = lme_signature(DATA / "longmemeval-zh/longmemeval_s_cleaned.json")
    zz, zzc = lme_signature(DATA / "longmemeval-zh/longmemeval_s_cleaned_zh.json")
    common = set(ee) & set(zz)
    removed = set(ee) - set(zz)
    added = set(zz) - set(ee)
    missing_sessions = 0
    session_mismatch = 0
    answer_source_missing = 0
    affected_questions = Counter()
    removed_sessions_by_type = Counter()
    for qid in common:
        en, zh = ee[qid], zz[qid]
        es, zs = en["sessions"], zh["sessions"]
        removed_here = len(set(es) - set(zs))
        missing_sessions += removed_here
        if removed_here:
            affected_questions[en["question_type"]] += 1
            removed_sessions_by_type[en["question_type"]] += removed_here
        session_mismatch += sum(es[s] != zs[s] for s in set(es) & set(zs))
        answer_source_missing += len(set(zh["answer_session_ids"]) - set(zs))
    lme = {
        "english": eec, "chinese": zzc,
        "common_question_ids": len(common), "removed_question_ids": len(removed),
        "added_question_ids": len(added),
        "removed_by_type": dict(Counter(ee[qid]["question_type"] for qid in removed)),
        "removed_haystack_sessions_within_common_questions": missing_sessions,
        "common_questions_with_haystack_sessions_removed": sum(affected_questions.values()),
        "affected_questions_by_type": dict(affected_questions),
        "removed_haystack_sessions_by_type": dict(removed_sessions_by_type),
        "added_haystack_sessions_within_common_questions": sum(len(set(zz[qid]["sessions"]) - set(ee[qid]["sessions"])) for qid in common),
        "shared_session_date_or_role_sequence_mismatches": session_mismatch,
        "shared_question_type_date_or_answer_session_mismatches": sum(
            any(ee[qid][k] != zz[qid][k] for k in ("question_type", "question_date", "answer_session_ids")) for qid in common),
        "chinese_answer_source_sessions_missing": answer_source_missing,
    }
    payload = {
        "scope": "Local dataset structure and benchmark files; no semantic translation review or model calls",
        "paths": {"locomo_en": str(ROOT / "data/locomo/locomo10.json"), "locomo_zh": str(DATA / "locomo-zh/locomo10_zh.json"),
                  "lme_en": str(DATA / "longmemeval-zh/longmemeval_s_cleaned.json"), "lme_zh": str(DATA / "longmemeval-zh/longmemeval_s_cleaned_zh.json")},
        "locomo": locomo, "longmemeval_s": lme,
        "historical_longmemeval_runs": longmemeval_runs(),
        "config": {
            "default_lme_points_to_english": "longmemeval_s_cleaned.json" in (ROOT / "config/default.yaml").read_text(),
            "locomo_zh_config_points_to_english_lme": "longmemeval_s_cleaned.json" in (ROOT / "config/locomo_zh.yaml").read_text(),
            "locomo_zh_config_path_exists": Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json").exists(),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"locomo": locomo, "longmemeval_s": lme, "config": payload["config"]}, ensure_ascii=False, indent=2))


def longmemeval_runs():
    runs = []
    for path in sorted((ROOT / "results").glob("*.jsonl")) + sorted((ROOT / "results_zh").glob("*.jsonl")):
        count, chinese_questions = 0, 0
        with path.open() as f:
            for line in f:
                if '"longmemeval_s"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("benchmark") != "longmemeval_s":
                    continue
                count += 1
                chinese_questions += any("\u4e00" <= ch <= "\u9fff" for ch in row.get("question", ""))
        if count:
            runs.append({"path": str(path.relative_to(ROOT)), "rows": count, "chinese_question_rows": chinese_questions})
    return runs


if __name__ == "__main__":
    main()
