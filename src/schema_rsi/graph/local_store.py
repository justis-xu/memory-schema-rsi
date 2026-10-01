"""本地 SQLite 图存储：与 HugeGraphStore 同接口的独立可回滚实现。

只实现建图与融合检索实际用到的面：create_schema / upsert_vertex / get_vertex /
upsert_edge / get_neighbors / count_vertices / reset_graph / _key_props。
检索行为是图内容的纯函数（同一 get_neighbors 契约），快照=单个 sqlite 文件可指纹。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from schema_rsi.graph.base import GraphSchema


class LocalGraphStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vertices ("
            "label TEXT, key TEXT, props TEXT, PRIMARY KEY (label, key))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS edges ("
            "label TEXT, out_label TEXT, out_key TEXT, in_label TEXT, in_key TEXT, "
            "props TEXT, PRIMARY KEY (label, out_label, out_key, in_label, in_key))"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_out ON edges (out_label, out_key)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_in ON edges (in_label, in_key)"
        )
        self._conn.commit()
        self._schema: dict | None = None
        self._key_props: dict[str, str] = {}

    # ---- schema ----
    def create_schema(self, schema: GraphSchema) -> None:
        for vl in schema.vertex_labels:
            self._key_props[vl.name] = vl.primary_key or "id"

    def schema_exists(self, schema: GraphSchema) -> bool:
        return bool(self._key_props)

    def drop_schema(self) -> None:
        self._schema, self._key_props = None, {}

    # ---- vertices ----
    def upsert_vertex(self, label: str, key: str, properties: dict | None = None) -> str:
        row = self._conn.execute(
            "SELECT props FROM vertices WHERE label=? AND key=?", (label, key)
        ).fetchone()
        if row:
            merged = json.loads(row[0])
            for k, v in (properties or {}).items():
                if v is not None:
                    merged[k] = str(v)
        else:
            merged = {self._key_props.get(label, "id"): str(key)}
            for k, v in (properties or {}).items():
                if v is not None:
                    merged[k] = str(v)
        self._conn.execute(
            "INSERT OR REPLACE INTO vertices (label, key, props) VALUES (?,?,?)",
            (label, key, json.dumps(merged, ensure_ascii=False)),
        )
        self._conn.commit()
        return f"{label}:{key}"

    def get_vertex(self, label: str, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT props FROM vertices WHERE label=? AND key=?", (label, key)
        ).fetchone()
        if not row:
            return None
        return {"id": f"{label}:{key}", "label": label, "properties": json.loads(row[0])}

    # ---- edges ----
    def upsert_edge(
        self,
        label: str,
        out_label: str,
        out_key: str,
        in_label: str,
        in_key: str,
        properties: dict | None = None,
    ) -> None:
        if self.get_vertex(out_label, out_key) is None or self.get_vertex(in_label, in_key) is None:
            raise KeyError(f"顶点不存在 out=({out_label},{out_key}) in=({in_label},{in_key})")
        self._conn.execute(
            "INSERT OR IGNORE INTO edges (label, out_label, out_key, in_label, in_key, props) "
            "VALUES (?,?,?,?,?,?)",
            (
                label, out_label, out_key, in_label, in_key,
                json.dumps(properties or {}, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def get_neighbors(
        self,
        label: str,
        key: str,
        direction: str = "BOTH",
        edge_labels: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        conds, params = [], []
        if direction == "OUT":
            conds.append("out_label=? AND out_key=?")
            params += [label, key]
            other_cols = ("in_label", "in_key")
        elif direction == "IN":
            conds.append("in_label=? AND in_key=?")
            params += [label, key]
            other_cols = ("out_label", "out_key")
        else:
            conds.append("(out_label=? AND out_key=?) OR (in_label=? AND in_key=?)")
            params += [label, key, label, key]
            other_cols = None
        q = f"SELECT * FROM edges WHERE {' AND '.join(conds) if other_cols else conds[0]}"
        # BOTH 时 OR 条件需整体括起
        if other_cols is None:
            q = f"SELECT * FROM edges WHERE ({conds[0]})"
        if edge_labels:
            q += f" AND label IN ({','.join('?' * len(edge_labels))})"
            params += edge_labels
        out = []
        seen: set[str] = set()
        for row in self._conn.execute(q, params):
            el, ol, ok_, il, ik, eprops = row
            if other_cols is None:
                if ol == label and ok_ == key:
                    lab, k = il, ik
                else:
                    lab, k = ol, ok_
            else:
                lab, k = (il, ik) if direction == "OUT" else (ol, ok_)
            if k in seen:
                continue
            v = self.get_vertex(lab, k)
            if v is None:
                continue
            seen.add(k)
            out.append({**v, "via_edge": el})
            if len(out) >= limit:
                break
        return out

    # ---- stats / maintenance ----
    def count_vertices(self, label: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM vertices WHERE label=?", (label,)
        ).fetchone()[0]

    def count_edges(self, label: str | None = None) -> int:
        if label:
            return self._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE label=?", (label,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    def reset_graph(self, *, drop_schema: bool = True) -> None:
        self._conn.execute("DELETE FROM edges")
        self._conn.execute("DELETE FROM vertices")
        self._conn.commit()
        if drop_schema:
            self.drop_schema()

    def backend_type(self) -> str:
        return "sqlite-local"

    def sha256(self) -> str:
        import hashlib

        self._conn.commit()
        return hashlib.sha256(self.path.read_bytes()).hexdigest()
