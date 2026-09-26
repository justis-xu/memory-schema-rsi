"""融合检索单测：build_concepts 用户隔离 + retrieve_fused 节点聚合/support/排除语义。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_rsi.evaluation.retriever import GraphRetriever
from schema_rsi.graph.builder import GraphBuilder
from schema_rsi.memory.base import MemoryRecord


class FakeGraphStore:
    """带邻接查询的假图存储（覆盖 _key_props / get_neighbors 语义）。"""

    _key_props = {"Memory": "memory_id", "Entity": "entity_id",
                  "Event": "event_id", "Preference": "pref_id", "Concept": "concept_id"}

    def __init__(self):
        self.vertices: dict[tuple[str, str], dict] = {}
        self.edges: list[tuple] = []  # (edge_label, out_label, out_key, in_label, in_key)

    def create_schema(self, schema):
        pass

    def upsert_vertex(self, label, key, properties=None):
        self.vertices[(label, key)] = dict(properties or {})

    def upsert_edge(self, label, out_label, out_key, in_label, in_key, properties=None):
        item = (label, out_label, out_key, in_label, in_key)
        if item not in self.edges:
            self.edges.append(item)

    def get_neighbors(self, label, key, direction="BOTH", edge_labels=None, limit=50):
        out = []
        for (el, ol, ok, il, ik) in self.edges:
            if edge_labels and el not in edge_labels:
                continue
            if direction in ("OUT", "BOTH") and (ol, ok) == (label, key):
                other = (il, ik)
            elif direction in ("IN", "BOTH") and (il, ik) == (label, key):
                other = (ol, ok)
            else:
                continue
            props = self.vertices.get(other, {})
            out.append({"label": other[0], "id": f"{other[0]}/{other[1]}",
                        "properties": props, "via_edge": el})
            if len(out) >= limit:
                break
        return out


def _mem(mid, owner="u1"):
    return MemoryRecord(id=mid, content=f"content of {mid}", metadata={"user_id": owner})


def test_build_concepts_scoped_and_edges():
    store = FakeGraphStore()
    builder = GraphBuilder(store)
    m1, m2, m3 = _mem("m1"), _mem("m2"), _mem("m3", owner="u2")
    report = builder.build_concepts([m1, m2, m3], {"m1": ["career"], "m2": ["career", "health"], "m3": ["career"]})

    # 同名主题按用户隔离：u1/career 与 u2/career 是两个 Concept 顶点
    concepts = [k for (lab, k) in store.vertices if lab == "Concept"]
    assert len(concepts) == 3  # u1:career, u1:health, u2:career
    assert report.label_counts["Concept"] == 3
    has_topic = [e for e in store.edges if e[0] == "HAS_TOPIC"]
    assert len(has_topic) == 4
    assert ("HAS_TOPIC", "Memory", "m3", "Concept", "2:u2:career") in has_topic


def test_retrieve_fused_aggregates_and_support():
    store = FakeGraphStore()
    for mid in ("m1", "m2", "m3", "m9"):
        store.upsert_vertex("Memory", mid, {"memory_id": mid, "user_id": "u1", "content": f"c{mid}"})
    store.upsert_vertex("Entity", "9:u1:doordash", {"entity_id": "9:u1:doordash", "name": "DoorDash"})
    store.upsert_vertex("Concept", "2:u1:career", {"concept_id": "2:u1:career", "name": "career"})
    store.upsert_edge("MENTIONS", "Memory", "m1", "Entity", "9:u1:doordash")
    store.upsert_edge("MENTIONS", "Memory", "m2", "Entity", "9:u1:doordash")
    store.upsert_edge("HAS_TOPIC", "Memory", "m1", "Concept", "2:u1:career")
    store.upsert_edge("HAS_TOPIC", "Memory", "m2", "Concept", "2:u1:career")
    store.upsert_edge("HAS_TOPIC", "Memory", "m3", "Concept", "2:u1:career")
    # 跨用户记忆挂在同主题节点上（Concept 主键带 owner，不会出现；这里构造跨用户 Memory 顶点验证防线）
    store.upsert_vertex("Memory", "mx", {"memory_id": "mx", "user_id": "u2", "content": "other user"})
    store.upsert_edge("HAS_TOPIC", "Memory", "mx", "Concept", "2:u1:career")

    retriever = GraphRetriever(store)
    seed = _mem("m1")
    out = retriever.retrieve_fused([seed], exclude_ids={"m1"}, per_node_limit=10, max_candidates=10)

    ids = [g["id"] for g in out]
    assert "m1" not in ids and "mx" not in ids  # 排除自身与跨用户
    assert set(ids) == {"m2", "m3"}
    by_id = {g["id"]: g for g in out}
    assert by_id["m2"]["support"] == 2  # entity + concept 两个节点拉回
    assert by_id["m3"]["support"] == 1
    assert sorted(by_id["m2"]["via"]) == ["concept:career", "entity:DoorDash"]
    # support 高者排前
    assert out[0]["id"] == "m2"


def test_retrieve_fused_respects_pool_cap():
    store = FakeGraphStore()
    store.upsert_vertex("Memory", "s", {"memory_id": "s", "user_id": "u1", "content": "seed"})
    store.upsert_vertex("Concept", "2:u1:hobby", {"concept_id": "2:u1:hobby", "name": "hobby"})
    for i in range(8):
        store.upsert_vertex("Memory", f"c{i}", {"memory_id": f"c{i}", "user_id": "u1", "content": f"c{i}"})
        store.upsert_edge("HAS_TOPIC", "Memory", f"c{i}", "Concept", "2:u1:hobby")
    store.upsert_edge("HAS_TOPIC", "Memory", "s", "Concept", "2:u1:hobby")

    out = GraphRetriever(store).retrieve_fused([_mem("s")], exclude_ids={"s"}, max_candidates=5)
    assert len(out) == 5


def test_rerank_pool_lexical_fallback():
    """rerank 关闭时池内排序走词面重叠（零 API），图相关候选能入选。"""
    from schema_rsi.config import get_settings
    from schema_rsi.evaluation.retriever import Mem0Retriever

    r = Mem0Retriever(backend=None, settings=get_settings())
    r._reranker = None  # 模拟 rerank 关闭

    def rec(mid, content):
        return MemoryRecord(id=mid, content=content, metadata={})

    pool = [
        rec("v1", "User enjoys hiking on weekends"),
        rec("v2", "DoorDash food delivery job started January"),
        rec("v3", "The DoorDash job was exhausting but paid well"),
        rec("v4", "Piano lessons every Tuesday evening"),
    ]
    out = r.rerank_pool("Where did the DoorDash delivery job happen?", pool, top_n=2)
    assert {x.id for x in out} == {"v2", "v3"}
