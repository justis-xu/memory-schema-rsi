"""Historical 2026-09-28 Mem0 cache SQL for reproducible offline audits.

The production SQLiteManager now uses insertion order to resolve timestamp
ties. This class preserves the prior behavior solely for in-memory replays.
Never use it for a real memory history database.
"""

import uuid
from datetime import datetime, timezone

from mem0.memory.storage import SQLiteManager


STORAGE_SHA256_BEFORE_FIX = "5f5ecdadd2b260fd14f36a0f338336eb760275465dd1056a99a47e1ef07702bd"


class LegacySQLiteManager(SQLiteManager):
    """Vendored storage.py's pre-fix save/get methods, isolated for audit."""

    def __new__(cls, db_path=":memory:"):
        if db_path != ":memory:":
            raise ValueError("historical cache replay is restricted to :memory:")
        return super().__new__(cls)

    def save_messages(self, messages, session_scope):
        if not messages:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                now = datetime.now(timezone.utc).isoformat()
                for message in messages:
                    self.connection.execute(
                        "INSERT INTO messages (id,session_scope,role,content,name,created_at) VALUES (?,?,?,?,?,?)",
                        (str(uuid.uuid4()), session_scope, message.get("role"), message.get("content"), message.get("name"), now),
                    )
                self.connection.execute("""DELETE FROM messages WHERE session_scope=? AND id NOT IN (
                    SELECT id FROM (SELECT id FROM messages WHERE session_scope=?
                                    ORDER BY created_at DESC LIMIT 10))""",
                                        (session_scope, session_scope))
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise

    def get_last_messages(self, session_scope, limit=10):
        with self._lock:
            rows = self.connection.execute("""SELECT role,content,name,created_at FROM (
                SELECT role,content,name,created_at FROM messages
                WHERE session_scope=? ORDER BY created_at DESC LIMIT ?)
                ORDER BY created_at ASC""", (session_scope, limit)).fetchall()
        return [{"role": row[0], "content": row[1], "name": row[2], "created_at": row[3]} for row in rows]
