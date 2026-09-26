"""GraphBuilder 单测：用 FakeGraphStore 验证 build/rebuild 语义（不依赖 HugeGraph）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_rsi.graph import GraphBuilder, make_test_schema
from schema_rsi.memory.base import MemoryRecord


class FakeGraphStore:
    def __init__(self):
        self.vertices: dict[tuple[str, str], dict] = {}
        self.edges: list[tuple] = []
        self.schema = None
        self.reset_count = 0

    def reset_graph(self, *, drop_schema: bool = True):
        self.vertices, self.edges, self.schema = {}, [], None
        self.reset_count += 1

    def create_schema(self, schema):
        self.schema = schema

    def upsert_vertex(self, label, key, properties=None):
        self.vertices[(label, key)] = dict(properties or {})
        return f"{label}/{key}"

    def upsert_edge(self, label, out_label, out_key, in_label, in_key, properties=None):
        self.edges.append((label, f"{out_label}/{out_key}", f"{in_label}/{in_key}", dict(properties or {})))

    def get_vertex(self, label, key):
        props = self.vertices.get((label, key))
        return {"label": label, "id": f"{label}/{key}", "properties": props or {}}


def make_memories() -> list[MemoryRecord]:
    def rec(i, session):
        return MemoryRecord(
            id=f"m{i}",
            content=f"memory {i}",
            metadata={"user_id": "u1", "session_id": session, "benchmark": "test"},
        )

    # 3 条 session=s1 + 2 条 session=s2（同 user）
    return [rec(1, "s1"), rec(2, "s1"), rec(3, "s1"), rec(4, "s2"), rec(5, "s2")]


def test_build_v0_strategy():
    store = FakeGraphStore()
    builder = GraphBuilder(store)
    report = builder.build(make_memories(), make_test_schema(), reset=True)

    # 每条 Memory 一个顶点
    assert report.vertices_written == 5
    assert len(store.vertices) == 5
    # 同 session 相邻建边: (3-1) + (2-1) = 3
    assert report.edges_written == 3
    assert len(store.edges) == 3
    # schema 驱动属性：只写 schema 定义过的属性名
    v = store.get_vertex("Memory", "m1")["properties"]
    assert v["memory_id"] == "m1" and v["content"] == "memory 1"
    assert "session_id" not in v  # 测试 schema 的 Memory 顶点没定义 session_id 属性
    assert store.reset_count == 1


def test_rebuild_resets():
    store = FakeGraphStore()
    builder = GraphBuilder(store)
    builder.build(make_memories(), make_test_schema())
    builder.rebuild(make_memories()[:2], make_test_schema())
    assert store.reset_count == 2
    assert len(store.vertices) == 2
    assert len(store.edges) == 1  # 同 session 相邻 2 条 -> 1 边


def test_schema_passed_through():
    store = FakeGraphStore()
    builder = GraphBuilder(store)
    schema = make_test_schema()
    builder.build(make_memories()[:1], schema)
    assert store.schema is schema
