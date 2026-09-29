"""Compare Mem0 get_all against the current local Chinese source snapshot.

Read-only enumeration. No extraction, embedding, answer, or graph calls.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.memory.mem0_backend import Mem0Backend, _GET_ALL_LIMIT


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUT = ROOT / "results/analysis/m2_current_source_enumeration_20260929.json"


def source_rows():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = conn.execute("""SELECT e.embedding_id,
            MAX(CASE WHEN m.key='data' THEN m.string_value END),
            MAX(CASE WHEN m.key='user_id' THEN m.string_value END),
            MAX(CASE WHEN m.key='session_id' THEN m.string_value END),
            MAX(CASE WHEN m.key='session_date' THEN m.string_value END)
            FROM embeddings e LEFT JOIN embedding_metadata m ON m.id=e.id
            GROUP BY e.id ORDER BY e.embedding_id""").fetchall()
    finally:
        conn.close()
    selected = [r for r in rows if str(r[2]).startswith("zhfull:locomo:")]
    digest = hashlib.sha256(json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return selected, digest


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    before_rows, before_sha = source_rows()
    users = sorted({r[2] for r in before_rows})
    assert len(users) == 10, "Current owner set changed; inspect before treating as prior cohort"
    backend = Mem0Backend(get_settings(str(ROOT / "config/locomo_zh.yaml")))
    reports = []
    for user_id in users:
        expected = {r[0]: r for r in before_rows if r[2] == user_id}
        actual_rows = backend.get_all_memories(user_id)
        actual = {r.id: r for r in actual_rows}
        duplicate_ids = len(actual_rows) - len(actual)
        absent = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        content_differences = sorted(i for i in set(expected) & set(actual)
                                     if expected[i][1] != actual[i].content)
        metadata_differences = sorted(i for i in set(expected) & set(actual)
                                      if (expected[i][2], expected[i][3], expected[i][4]) !=
                                      (actual[i].metadata.get("user_id"),
                                       actual[i].metadata.get("session_id"),
                                       actual[i].metadata.get("session_date")))
        reports.append({
            "user_id": user_id, "sqlite_count": len(expected), "get_all_count": len(actual_rows),
            "below_get_all_limit": len(expected) < _GET_ALL_LIMIT,
            "duplicate_returned_ids": duplicate_ids, "missing_ids": absent, "extra_ids": extra,
            "content_mismatch_ids": content_differences,
            "user_session_date_mismatch_ids": metadata_differences,
            "matches_current_sqlite": not any((duplicate_ids, absent, extra,
                                                content_differences, metadata_differences)),
        })
    after_rows, after_sha = source_rows()
    result = {
        "scope": "Current local zhfull:locomo: 10-user Chroma snapshot only; configured Mem0Backend.get_all_memories versus direct SQLite ID/content/user/session/date enumeration; no source or graph writes.",
        "db_path": str(DB.relative_to(ROOT)),
        "get_all_limit": _GET_ALL_LIMIT,
        "logical_source_sha256_before": before_sha,
        "logical_source_sha256_after": after_sha,
        "source_unchanged_during_audit": before_sha == after_sha and before_rows == after_rows,
        "users": reports,
        "totals": {
            "users": len(users), "sqlite_memories": len(before_rows),
            "get_all_memories": sum(r["get_all_count"] for r in reports),
            "users_matching_sqlite": sum(r["matches_current_sqlite"] for r in reports),
        },
        "complete_for_current_local_snapshot": before_sha == after_sha and before_rows == after_rows and
                                               all(r["matches_current_sqlite"] and r["below_get_all_limit"] for r in reports),
        "limits": [
            "A before/after logical fingerprint does not lock out concurrent writes between the two reads.",
            "The SQLite source is a current local snapshot, not a frozen historical ingest snapshot.",
            "ID and text completeness do not establish structured extraction success or graph relationship provenance.",
            "get_all excludes expired records by default; equality here applies to the currently visible records.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**result["totals"], "source_unchanged": result["source_unchanged_during_audit"],
                      "complete_for_current_local_snapshot": result["complete_for_current_local_snapshot"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
