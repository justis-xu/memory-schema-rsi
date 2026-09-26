#!/usr/bin/env python3
"""Audit gold source-session reachability for Chinese LoCoMo cases.

Session presence is only a coarse diagnostic. It does not establish that a
specific gold turn or its fact survived extraction into a memory.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
RAW = ROOT / "results_zh/zhfull_locomo_20260925_115806.jsonl"
GRADES = ROOT / "results/precision_regrade_zhfull.jsonl"
STORE = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUTPUT = ROOT / "results/analysis/evidence_session_audit_20260926.json"
CATEGORIES = {"single-hop", "multi-hop", "temporal", "open-domain"}
STABLE_CONVS = {"conv-26", "conv-30", "conv-43", "conv-44"}


def jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if row["case_id"] in rows:
            raise ValueError(f"duplicate case ID: {row['case_id']}")
        rows[row["case_id"]] = row
    return rows


def current_pool() -> tuple[dict[str, set[str]], dict[str, dict[str, str | None]]]:
    id_by_user: dict[str, set[str]] = defaultdict(set)
    session_by_id: dict[str, dict[str, str | None]] = defaultdict(dict)
    with sqlite3.connect(f"file:{STORE}?mode=ro", uri=True) as db:
        metadata: dict[int, dict[str, str]] = defaultdict(dict)
        for internal_id, key, value in db.execute(
            "SELECT id, key, string_value FROM embedding_metadata WHERE key IN ('user_id', 'session_id')"
        ):
            metadata[internal_id][key] = value
        for internal_id, memory_id in db.execute("SELECT id, embedding_id FROM embeddings"):
            info = metadata[internal_id]
            if info.get("user_id"):
                id_by_user[info["user_id"]].add(memory_id)
                session_by_id[info["user_id"]][memory_id] = info.get("session_id")
    return id_by_user, session_by_id


def main() -> None:
    dataset = json.loads(DATASET.read_text())
    questions = {
        f"locomo_{entry['sample_id']}_qa{index}": qa
        for entry in dataset for index, qa in enumerate(entry["qa"])
    }
    raw, grades = jsonl(RAW), jsonl(GRADES)
    pool_ids, session_by_id = current_pool()
    archived_ids = {}
    for conv in STABLE_CONVS:
        user = f"zhfull:locomo:{conv}"
        archived_ids[user] = {
            memory["id"] for row in raw.values() if row["metadata"].get("user_id") == user
            for memory in row.get("retrieved_memories", [])
        }
        if archived_ids[user] - pool_ids[user]:
            raise ValueError(f"historical IDs no longer in current pool: {conv}")
    summary: dict[str, Counter] = defaultdict(Counter)
    by_category: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    by_conversation: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    cases = []
    for case_id, row in raw.items():
        match = re.search(r"conv-\d+", case_id)
        assert match, case_id
        conv = match.group(0)
        if conv not in STABLE_CONVS or grades[case_id]["category"] not in CATEGORIES:
            continue
        gold = {
            f"session_{m.group(1)}"
            for token in questions[case_id].get("evidence") or []
            for m in re.finditer(r"D:?(\d+):\d+", str(token))
        }
        if not gold:
            continue
        user = f"zhfull:locomo:{conv}"
        records = {m["id"]: m for m in row.get("retrieved_memories", []) + row.get("graph_memories", [])}
        top_ids = row["metadata"]["answer_context_ids"]
        missing_records = [memory_id for memory_id in top_ids if memory_id not in records]
        if missing_records:
            raise ValueError(f"missing answer context records: {case_id}, {missing_records}")
        top_sessions = {(records[memory_id].get("metadata") or {}).get("session_id") for memory_id in top_ids}
        archived_pool_sessions = {session_by_id[user][memory_id] for memory_id in archived_ids[user]}
        stage = "all_gold_sessions_in_top15" if gold <= top_sessions else (
            "all_gold_sessions_in_pool_only" if gold <= archived_pool_sessions else "some_gold_session_absent_from_pool"
        )
        category = grades[case_id]["category"]
        exact = grades[case_id]["final"] == "exact"
        summary[stage]["cases"] += 1
        summary[stage]["exact"] += exact
        by_category[category][stage]["cases"] += 1
        by_category[category][stage]["exact"] += exact
        by_conversation[conv][stage]["cases"] += 1
        by_conversation[conv][stage]["exact"] += exact
        cases.append({
            "case_id": case_id, "category": category, "strict_final": grades[case_id]["final"],
            "gold_sessions": sorted(gold), "top15_sessions": sorted(s for s in top_sessions if s),
            "pool_missing_gold_sessions": sorted(gold - archived_pool_sessions), "stage": stage,
        })
    id_check = {}
    for conv in sorted(STABLE_CONVS):
        user = f"zhfull:locomo:{conv}"
        archived = archived_ids[user]
        id_check[conv] = {
            "archived_retrieved_unique": len(archived),
            "archived_retrieved_in_current_pool": len(archived & pool_ids[user]),
            "current_pool_size": len(pool_ids[user]),
        }
    report = {
        "scope": "Four conversations with all historical retrieved IDs still in current Chroma; J categories with parseable gold Dn:turn evidence. Pool is restricted to archived retrieved IDs to exclude later additions.",
        "warning": "Presence of session_n is not evidence that the specific gold turn or fact was extracted. Archived retrieved-ID union is used as a pool proxy and may omit unqueried memories.",
        "rows": len(cases), "id_check": id_check,
        "stages": {k: dict(v) for k, v in sorted(summary.items())},
        "by_category": {k: {s: dict(c) for s, c in sorted(v.items())} for k, v in sorted(by_category.items())},
        "by_conversation": {k: {s: dict(c) for s, c in sorted(v.items())} for k, v in sorted(by_conversation.items())},
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "id_check", "stages", "by_category")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
