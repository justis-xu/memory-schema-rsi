"""Regression tests for the vendored Mem0 recent-message cache."""

from datetime import datetime, timedelta, timezone

import mem0.memory.storage as storage


def _contents(db, scope):
    return [row["content"] for row in db.get_last_messages(scope, limit=10)]


def test_long_batch_retains_latest_ten_in_conversation_order():
    db = storage.SQLiteManager(":memory:")
    try:
        messages = [{"role": "system", "content": "date anchor"}] + [
            {"role": "user", "content": f"turn-{i}"} for i in range(1, 25)
        ]
        db.save_messages(messages, "user_id=one")
        assert _contents(db, "user_id=one") == [f"turn-{i}" for i in range(15, 25)]
    finally:
        db.connection.close()


def test_equal_timestamps_across_batches_still_keep_latest_insertions(monkeypatch):
    real_datetime = datetime

    class FrozenDateTime:
        @staticmethod
        def now(tz):
            return real_datetime(2026, 9, 28, tzinfo=timezone.utc)

    monkeypatch.setattr(storage, "datetime", FrozenDateTime)
    db = storage.SQLiteManager(":memory:")
    try:
        db.save_messages([{"role": "user", "content": f"old-{i}"} for i in range(8)], "user_id=one")
        db.save_messages([{"role": "user", "content": f"new-{i}"} for i in range(5)], "user_id=one")
        assert _contents(db, "user_id=one") == [f"old-{i}" for i in range(3, 8)] + [
            f"new-{i}" for i in range(5)
        ]
    finally:
        db.connection.close()


def test_recent_cache_is_isolated_by_scope(monkeypatch):
    real_datetime = datetime
    counter = iter(range(3))

    class AdvancingDateTime:
        @staticmethod
        def now(tz):
            return real_datetime(2026, 9, 28, tzinfo=timezone.utc) + timedelta(microseconds=next(counter))

    monkeypatch.setattr(storage, "datetime", AdvancingDateTime)
    db = storage.SQLiteManager(":memory:")
    try:
        db.save_messages([{"role": "user", "content": f"A-{i}"} for i in range(12)], "user_id=A")
        db.save_messages([{"role": "assistant", "content": "B-only"}], "user_id=B")
        db.save_messages([{"role": "user", "content": "A-latest"}], "user_id=A")
        assert _contents(db, "user_id=A") == [f"A-{i}" for i in range(3, 12)] + ["A-latest"]
        assert _contents(db, "user_id=B") == ["B-only"]
    finally:
        db.connection.close()
