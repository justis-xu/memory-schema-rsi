"""Audit literal date-string sorting in archived answer contexts, no model calls."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from schema_rsi.benchmarks.base import parse_session_date  # noqa: E402


FILES = {
    "zh_locomo": ROOT / "results_zh/zhfull_locomo_20260925_164645.jsonl",
    "en_locomo": ROOT / "results/full_locomo_finalC_ctrl_20260925_090746.jsonl",
    "en_lme_small": ROOT / "results/full_lme_20260924_013242.jsonl",
}
CURRENT = ROOT / "results/analysis/m2_current_zh_stratified_retrieval_20260927.json"
OUT = ROOT / "results/analysis/m2_prompt_date_order_20260927.json"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_context(case_id, category, records, *, newest_first=False):
    dates = [r.get("session_date") for r in records]
    parsed = [parse_session_date(date) for date in dates]
    valid = [date for date in dates if parse_session_date(date)]
    lex = sorted(valid, reverse=newest_first)
    actual = sorted(valid, key=parse_session_date, reverse=newest_first)
    pairs = inversions = 0
    for i, a in enumerate(lex):
        for b in lex[i + 1:]:
            da, db = parse_session_date(a), parse_session_date(b)
            if da == db:
                continue
            pairs += 1
            inversions += (da < db if newest_first else da > db)
    return {
        "case_id": case_id, "category": category,
        "memory_count": len(records), "dated_count": len(valid),
        "unparseable_or_missing_count": len(dates) - len(valid),
        "distinct_dates": len(set(parsed) - {None}),
        "chronological_order_wrong": lex != actual,
        "inverted_date_pairs": inversions,
        "comparable_date_pairs": pairs,
    }


def audit_file(path, newest_first=False):
    cases = []
    context_mismatch = []
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        memories = row.get("retrieved_memories") or []
        ids = row.get("metadata", {}).get("answer_context_ids") or []
        lookup = {m["id"]: m for m in memories}
        if len(ids) != len(memories) or any(mid not in lookup for mid in ids):
            context_mismatch.append(row["case_id"])
            continue
        records = [(lookup[mid].get("metadata") or {}) for mid in ids]
        cases.append(audit_context(row["case_id"], row.get("metadata", {}).get("category"),
                                   records, newest_first=newest_first))
    return summarize(cases, context_mismatch)


def summarize(cases, context_mismatch=None):
    eligible = [c for c in cases if c["distinct_dates"] >= 2]
    changed = [c for c in eligible if c["chronological_order_wrong"]]
    by_category = {}
    for cat in sorted({str(c["category"]) for c in cases}):
        group = [c for c in eligible if str(c["category"]) == cat]
        by_category[cat] = {"eligible": len(group),
                            "wrong_order": sum(c["chronological_order_wrong"] for c in group)}
    return {
        "cases_read": len(cases) + len(context_mismatch or []),
        "contexts_reconstructable": len(cases),
        "contexts_not_reconstructable": context_mismatch or [],
        "eligible_two_distinct_dates": len(eligible),
        "wrong_chronological_order": len(changed),
        "comparable_date_pairs": sum(c["comparable_date_pairs"] for c in eligible),
        "inverted_date_pairs": sum(c["inverted_date_pairs"] for c in eligible),
        "unparseable_or_missing_memory_dates": sum(c["unparseable_or_missing_count"] for c in cases),
        "by_category": by_category,
        "examples": sorted(changed, key=lambda c: c["inverted_date_pairs"], reverse=True)[:5],
    }


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    result = {
        "scope": "Archived no-graph answer contexts and frozen current Chinese retrieval cohort; compares raw-string date order with parsed chronological order, no model calls",
        "files": {},
    }
    for name, path in FILES.items():
        result["files"][name] = {"path": str(path.relative_to(ROOT)),
                                 "sha256": file_hash(path),
                                 "audit": audit_file(path, newest_first=name == "en_lme_small")}
    current = json.loads(CURRENT.read_text())
    cases = [audit_context(c["case_id"], c["category"],
                           c["reranked"], newest_first=False) for c in current["cases"]]
    result["current_zh_20"] = {"path": str(CURRENT.relative_to(ROOT)),
                                "sha256": file_hash(CURRENT), "audit": summarize(cases)}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, entry in result["files"].items():
        print(name, {k: entry["audit"][k] for k in
                     ("cases_read", "contexts_reconstructable", "eligible_two_distinct_dates",
                      "wrong_chronological_order", "inverted_date_pairs", "comparable_date_pairs")})
    print("current_zh_20", {k: result["current_zh_20"]["audit"][k] for k in
                            ("eligible_two_distinct_dates", "wrong_chronological_order",
                             "inverted_date_pairs", "comparable_date_pairs")})


if __name__ == "__main__":
    main()
