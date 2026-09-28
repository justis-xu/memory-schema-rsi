"""Replay Mem0's ten-message cache on Chinese LoCoMo input, without a model.

Uses the installed vendored SQLiteManager in isolated in-memory databases.
The fixed SQL comparison is a diagnostic prototype; production code is not edited.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from mem0.memory.storage import SQLiteManager

from schema_rsi.benchmarks.base import parse_session_date
from schema_rsi.benchmarks.locomo import LocomoDataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
HISTORY = ROOT / "data/vector_store_zh/mem0_history_zh.db"
OUTPUT = ROOT / "results/analysis/m2_last_messages_tie_20260928.json"
TARGETS = {
    "conv-26": {"source_session": "session_18", "written_session": "session_19",
                "source_dia_ids": ["D18:1", "D18:3", "D18:7"],
                "older_ids": ["7da557c9-eb9c-4392-b9c5-2ac9863d2695", "2ec90c89-e8d4-4fd1-80bd-2eca86fb2c2f"],
                "later_id": "f87df58f-81e4-46e4-a2bc-22e62b98c625"},
    "conv-41": {"source_session": "session_31", "written_session": "session_32",
                "source_dia_ids": ["D31:1", "D31:9"],
                "older_ids": ["21519f92-6587-4616-926b-120da1779306", "0467de53-751f-4955-9bf3-5b19ded453cc"],
                "later_id": "3e495438-a164-4250-ab0f-9f479a67a7dc"},
}


def batch_messages(session):
    messages = [{"role": turn["role"], "content": turn["content"]} for turn in session["turns"]]
    date = parse_session_date(session.get("date"))
    if date:
        messages.insert(0, {"role": "system", "content": (
            f"Conversation date: {date}. Use this date to resolve relative time expressions like "
            "'yesterday', 'next month', 'in three weeks'.")})
    return messages


def fixed_sql_probe(length):
    """Compare old/fixed ORDER BY with a single batch timestamp."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, session_scope TEXT, role TEXT, content TEXT, name TEXT, created_at DATETIME)")
    conn.executemany("INSERT INTO messages VALUES (?, 'scope', 'user', ?, NULL, ?)",
                     [(f"old-{i}", f"old-{i}", "2026-01-01T00:00:00Z") for i in range(10)])
    conn.executemany("INSERT INTO messages VALUES (?, 'scope', 'user', ?, NULL, ?)",
                     [(str(i), str(i), "2026-01-01T00:00:01Z") for i in range(length)])
    before = [int(row[0]) for row in conn.execute(
        "SELECT id FROM messages WHERE session_scope='scope' ORDER BY created_at DESC LIMIT 10")]
    after = [int(row[0]) for row in conn.execute(
        "SELECT id FROM messages WHERE session_scope='scope' ORDER BY created_at DESC, rowid DESC LIMIT 10")]
    conn.close()
    return {"current_order_selected_positions": sorted(before), "rowid_tiebreak_selected_positions": sorted(after)}


def history_evidence():
    conn = sqlite3.connect(f"file:{HISTORY}?mode=ro", uri=True)
    try:
        out = {}
        for conv, spec in TARGETS.items():
            ids = spec["older_ids"] + [spec["later_id"]]
            rows = [conn.execute("SELECT memory_id,event,created_at,new_memory FROM history WHERE memory_id=? AND event='ADD'", (mid,)).fetchall()
                    for mid in ids]
            assert all(len(group) == 1 for group in rows), (conv, rows)
            out[conv] = [{"memory_id": row[0][0], "event": row[0][1], "created_at": row[0][2], "content": row[0][3]} for row in rows]
            assert all(row["created_at"] < out[conv][-1]["created_at"] for row in out[conv][:-1])
        return out, conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()


def main():
    dataset = LocomoDataset(path=SOURCE)
    cases = {case.metadata["conversation_id"]: case for case in dataset.cases if case.case_id.endswith("_qa0")}
    total = 0
    long_sessions = 0
    first_ten = 0
    last_ten = 0
    target_replays = {}
    for conv, case in sorted(cases.items()):
        cache = SQLiteManager(":memory:")
        scope = f"user_id=zhfull:locomo:{conv}"
        for session in case.history:
            total += 1
            messages = batch_messages(session)
            cache.save_messages(messages, scope)
            retained = cache.get_last_messages(scope, limit=10)
            contents = [m["content"] for m in retained]
            expected_first = [m["content"] for m in messages[:10]]
            expected_last = [m["content"] for m in messages[-10:]]
            if len(messages) > 10:
                long_sessions += 1
                first_ten += contents == expected_first
                last_ten += contents == expected_last
            if conv in TARGETS and session["session_id"] == TARGETS[conv]["source_session"]:
                dia_by_content = {turn["content"]: turn["dia_id"] for turn in session["turns"]}
                retained_dia_ids = [dia_by_content.get(value, "date_anchor_or_other") for value in contents]
                spec = TARGETS[conv]
                assert set(spec["source_dia_ids"]).issubset(retained_dia_ids)
                target_replays[conv] = {
                    "source_session": session["session_id"], "next_written_session": spec["written_session"],
                    "batch_message_count": len(messages), "retained_dia_ids_after_source_session": retained_dia_ids,
                    "source_dia_ids_present": spec["source_dia_ids"],
                    "fixed_sql": fixed_sql_probe(len(messages)),
                }
        cache.connection.close()
    assert set(target_replays) == set(TARGETS)
    history, surviving_message_rows = history_evidence()
    payload = {
        "scope": "Offline replay of actual SQLiteManager.save_messages/get_last_messages over all 10 Chinese LoCoMo conversations; no LLM, no real database writes",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "history_db_sha256": hashlib.sha256(HISTORY.read_bytes()).hexdigest(),
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            ROOT / "third_party/mem0-src/mem0/memory/storage.py",
            ROOT / "third_party/mem0-src/mem0/memory/main.py",
            ROOT / "src/schema_rsi/benchmarks/locomo.py",
            ROOT / "src/schema_rsi/evaluation/pipeline.py")},
        "counts": {"conversations": len(cases), "sessions": total, "batches_longer_than_ten_messages": long_sessions,
                   "first_ten_retained": first_ten, "last_ten_retained": last_ten,
                   "current_history_surviving_message_rows": surviving_message_rows},
        "target_replays": target_replays, "history_add_records": history,
        "limits": ["重放的是当前源码和同版中文数据，不是2026-09-25提取调用的原始请求/回复",
                   "旧库仅保留每owner最近10条消息，history记录ADD但不保存实际Existing Memories检索或LLM输出",
                   "rowid并列排序原型只验证缓存位置变化；Existing Memories通道仍可能让旧事实进入提取提示"],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
