"""GraphBuilder：Memory（事实源）→ Graph（派生产物）。

核心契约：
    Memory = Source of Truth
    Graph  = Derived Artifact（随时 reset 并按 Memory + Schema 重建）

未来 SchemaRSI 会针对同一批 Memory 不断产出 Schema S0/S1/S2... 并用
GraphBuilder.build(memories, schema) 重建图，本接口即为此预留。

⚠️ V0 构图策略【不是正式 Graph Schema 设计】，仅 infrastructure smoke test：
    - 每条 Memory → schema 主顶点类型的一个顶点（业务主键 = memory id）
    - 同一 (user_id, session_id) 分组内相邻两条 Memory → 第一条边类型的边
      （测试 Schema 里即 RELATED_TO，语义是"同会话相邻"，不代表真实关系）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from schema_rsi.graph.schema import GraphSchema
from schema_rsi.memory.base import MemoryRecord


@dataclass
class GraphReport:
    schema_name: str
    vertices_written: int = 0
    edges_written: int = 0
    duration_s: float = 0.0
    label_counts: dict = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _scoped_key(owner: str, name: str) -> str:
    normalized = " ".join(str(name).lower().strip().split())
    return f"{len(owner)}:{owner}:{normalized}" if normalized else ""


class GraphBuilder:
    def __init__(self, store):
        self.store = store

    def build(self, memories: list[MemoryRecord], schema: GraphSchema, *, reset: bool = True) -> GraphReport:
        if any(not (rec.metadata or {}).get("user_id") for rec in memories):
            raise ValueError("构图要求每条记忆都有 user_id")
        report = GraphReport(schema_name=schema.name)
        t0 = time.perf_counter()

        if reset:
            self.store.reset_graph(drop_schema=True)
        self.store.create_schema(schema)

        vlabel = schema.primary_vertex_label()
        if vlabel is None:
            raise ValueError(f"schema '{schema.name}' 没有任何 vertex label")
        elabel = schema.edge_labels[0] if schema.edge_labels else None
        key_prop = vlabel.primary_key or "memory_id"
        prop_names = set(vlabel.property_names())
        edge_prop_names = set(elabel.property_names()) if elabel else set()

        # 保序分组：(user_id, session_id) —— 仅 smoke 用策略
        groups: dict[tuple[str, str], list[MemoryRecord]] = {}
        for rec in memories:
            gk = (str(rec.metadata.get("user_id", "")), str(rec.metadata.get("session_id", "")))
            groups.setdefault(gk, []).append(rec)

        for (user_id, session_id), recs in groups.items():
            prev_key: str | None = None
            for rec in recs:
                props = self._vertex_props(rec, prop_names, key_prop)
                self.store.upsert_vertex(vlabel.name, rec.id, props)
                report.vertices_written += 1
                if elabel is not None and prev_key is not None:
                    eprops: dict[str, str] = {}
                    if "reason" in edge_prop_names:
                        eprops["reason"] = "adjacent_in_session"
                    if "session_id" in edge_prop_names:
                        eprops["session_id"] = session_id
                    if "user_id" in edge_prop_names:
                        eprops["user_id"] = user_id
                    self.store.upsert_edge(
                        elabel.name, vlabel.name, prev_key, vlabel.name, rec.id, eprops
                    )
                    report.edges_written += 1
                prev_key = rec.id

        report.duration_s = round(time.perf_counter() - t0, 3)
        report.label_counts = {vlabel.name: report.vertices_written}
        return report

    @staticmethod
    def _vertex_props(rec: MemoryRecord, prop_names: set[str], key_prop: str) -> dict[str, str]:
        """把 MemoryRecord 映射为 schema 定义的顶点属性（schema 驱动，未定义的属性丢弃）。"""
        meta = rec.metadata or {}
        source: dict[str, str] = {
            key_prop: str(rec.id),
            "memory_id": str(rec.id),
            "content": rec.content,
        }
        for name in ("user_id", "session_id", "benchmark", "case_id", "created_at"):
            if name in meta and meta[name] is not None:
                source[name] = str(meta[name])
        source.setdefault("user_id", str(meta.get("user_id", "")))
        source.setdefault("source", str(meta.get("benchmark", "mem0")))
        source.setdefault("created_at", _now_iso())
        return {k: v for k, v in source.items() if k in prop_names}

    def rebuild(self, memories: list[MemoryRecord], schema: GraphSchema) -> GraphReport:
        """reset + build 的显式别名，供 Schema 演化实验调用。"""
        return self.build(memories, schema, reset=True)

    # ---- Schema S1：Memory + Entity + Event + Preference + 多种边 ----

    def build_s1(self, memories: list[MemoryRecord], extraction: dict[str, dict], schema=None, *, reset: bool = True) -> GraphReport:
        """按 Schema S1 建图：在 S0 基础上增加 Event/Preference 顶点和多类边。

        extraction 来自 StructuredExtractor（含 entities/events/preferences/relationships）。
        """
        from schema_rsi.graph.schema import make_schema_s1

        if any(not (rec.metadata or {}).get("user_id") for rec in memories):
            raise ValueError("S1 构图要求每条记忆都有 user_id")
        if memories and not any(
            any(extraction.get(rec.id, {}).get(kind) for kind in
                ("entities", "events", "preferences", "relationships"))
            for rec in memories
        ):
            raise ValueError("S1 结构化提取全为空；拒绝清空现有图")
        schema = schema or make_schema_s1()
        report = GraphReport(schema_name=schema.name)
        t0 = time.perf_counter()

        if reset:
            self.store.reset_graph(drop_schema=True)
        self.store.create_schema(schema)

        mem_label = schema.vertex_label("Memory")
        ent_label = schema.vertex_label("Entity")
        evt_label = schema.vertex_label("Event")
        pref_label = schema.vertex_label("Preference")

        # 1) Memory 顶点
        for rec in memories:
            props = self._vertex_props(rec, set(mem_label.property_names()), "memory_id")
            if "session_id" in mem_label.property_names():
                props["session_id"] = str(rec.metadata.get("session_id", ""))
            if "session_date" in mem_label.property_names():
                props["session_date"] = str(rec.metadata.get("session_date") or "")
            self.store.upsert_vertex("Memory", rec.id, props)
            report.vertices_written += 1

        # 2) Entity 顶点 + MENTIONS + RELATED_TO
        entities: dict[str, dict] = {}
        entity_relations: list[tuple] = []  # (from_key, relation, to_key)
        for rec in memories:
            ext = extraction.get(rec.id, {})
            owner = str(rec.metadata["user_id"])
            for ent in ext.get("entities", []):
                key = _scoped_key(owner, ent.get("name", ""))
                if not key:
                    continue
                if key not in entities:
                    props = {"entity_id": key, "name": ent["name"],
                             "type": ent.get("type", "other")}
                    if ent.get("date"):
                        props["date"] = str(ent["date"])
                    self.store.upsert_vertex("Entity", key, props)
                    entities[key] = props
                    report.vertices_written += 1
                self.store.upsert_edge("MENTIONS", "Memory", rec.id, "Entity", key)
                report.edges_written += 1

            for rel in ext.get("relationships", []):
                fk = _scoped_key(owner, rel.get("from_entity", ""))
                tk = _scoped_key(owner, rel.get("to_entity", ""))
                if fk and tk:
                    entity_relations.append((fk, rel.get("relation", "related"), tk))

        # RELATED_TO 边（Entity-Entity）
        for fk, relation, tk in set(entity_relations):
            if fk in entities and tk in entities and fk != tk:
                self.store.upsert_edge("RELATED_TO", "Entity", fk, "Entity", tk,
                                       {"relation": relation})
                report.edges_written += 1

        # 3) Event 顶点 + DESCRIBES_EVENT + PARTICIPATES_IN
        events: dict[str, dict] = {}
        for rec in memories:
            ext = extraction.get(rec.id, {})
            owner = str(rec.metadata["user_id"])
            for evt in ext.get("events", []):
                name = evt.get("name", "").strip()
                if not name:
                    continue
                date = evt.get("date", "").strip()
                ekey = _scoped_key(owner, f"{name}|{date}")  # 仅同用户同名同时期合并
                if ekey not in events:
                    props = {"event_id": ekey, "name": name, "date": date or "unknown",
                             "participants": evt.get("participants", ""),
                             "description": name}
                    self.store.upsert_vertex("Event", ekey, props)
                    events[ekey] = props
                    report.vertices_written += 1
                self.store.upsert_edge("DESCRIBES_EVENT", "Memory", rec.id, "Event", ekey)
                report.edges_written += 1
                # 参与者 → PARTICIPATES_IN
                for p in str(evt.get("participants", "")).split(","):
                    pk = _scoped_key(owner, p)
                    if pk and pk in entities:
                        self.store.upsert_edge("PARTICIPATES_IN", "Entity", pk, "Event", ekey)
                        report.edges_written += 1

        # 4) Preference 顶点 + EXPRESSES_PREFERENCE + HAS_PREFERENCE
        prefs: dict[str, dict] = {}
        for rec in memories:
            ext = extraction.get(rec.id, {})
            owner = str(rec.metadata["user_id"])
            for pref in ext.get("preferences", []):
                subject = pref.get("subject", "").strip()
                obj = pref.get("object", "").strip()
                if not subject or not obj:
                    continue
                ptype = pref.get("pref_type", "prefer").strip().lower()
                pkey = _scoped_key(owner, f"{subject}|{ptype}|{obj}")
                if pkey not in prefs:
                    props = {"pref_id": pkey, "subject": subject,
                             "pref_type": ptype, "object": obj}
                    self.store.upsert_vertex("Preference", pkey, props)
                    prefs[pkey] = props
                    report.vertices_written += 1
                self.store.upsert_edge("EXPRESSES_PREFERENCE", "Memory", rec.id, "Preference", pkey)
                report.edges_written += 1
                sk = _scoped_key(owner, subject)
                if sk in entities:
                    self.store.upsert_edge("HAS_PREFERENCE", "Entity", sk, "Preference", pkey)
                    report.edges_written += 1

        report.duration_s = round(time.perf_counter() - t0, 3)
        report.label_counts = {
            "Memory": len(memories),
            "Entity": len(entities),
            "Event": len(events),
            "Preference": len(prefs),
        }
        return report

    def build_concepts(self, memories: list[MemoryRecord], topics: dict[str, list[str]], schema=None) -> GraphReport:
        """在已有 S1 图上增量写入 Concept 顶点与 HAS_TOPIC 边（不 reset）。

        topics 来自 TopicExtractor（{memory_id: [topic, ...]}，受控词表）。
        Concept 主键沿用用户隔离规则 _scoped_key(owner, topic)。
        """
        from schema_rsi.graph.schema import make_schema_s2

        schema = schema or make_schema_s2()
        self.store.create_schema(schema)  # 幂等：只补缺失的 Concept/HAS_TOPIC
        report = GraphReport(schema_name=schema.name)
        t0 = time.perf_counter()

        concepts: dict[str, dict] = {}
        for rec in memories:
            owner = str((rec.metadata or {}).get("user_id") or "")
            if not owner:
                continue
            for topic in topics.get(rec.id, []):
                ckey = _scoped_key(owner, topic)
                if not ckey:
                    continue
                if ckey not in concepts:
                    self.store.upsert_vertex(
                        "Concept", ckey, {"concept_id": ckey, "name": topic, "user_id": owner}
                    )
                    concepts[ckey] = {"name": topic}
                    report.vertices_written += 1
                self.store.upsert_edge("HAS_TOPIC", "Memory", rec.id, "Concept", ckey)
                report.edges_written += 1

        report.duration_s = round(time.perf_counter() - t0, 3)
        report.label_counts = {"Concept": len(concepts)}
        return report

    def build_s0(self, memories: list[MemoryRecord], extraction: dict[str, list[dict]], schema=None, *, reset: bool = True) -> GraphReport:
        """按 Schema S0 建图：Memory 顶点 + Entity 顶点 + Memory-[MENTIONS]->Entity。

        extraction: {memory_id: [{"name","type","date"?}]}（来自 EntityExtractor，可缓存）。
        Entity 的业务主键包含 user_id；只合并同一用户的同名实体。
        """
        if any(not (rec.metadata or {}).get("user_id") for rec in memories):
            raise ValueError("S0 构图要求每条记忆都有 user_id")
        if memories and not any(extraction.get(rec.id) for rec in memories):
            raise ValueError("S0 实体提取全为空；拒绝清空现有图")
        schema = schema or make_schema_s0_import()
        report = GraphReport(schema_name=schema.name)
        t0 = time.perf_counter()

        if reset:
            self.store.reset_graph(drop_schema=True)
        self.store.create_schema(schema)

        mem_label = schema.vertex_label("Memory")
        ent_label = schema.vertex_label("Entity")
        mem_props = set(mem_label.property_names())
        ent_props = set(ent_label.property_names())

        # 1) Memory 顶点
        for rec in memories:
            props = self._vertex_props(rec, mem_props, mem_label.primary_key or "memory_id")
            if "session_id" in mem_props:
                props["session_id"] = str(rec.metadata.get("session_id", ""))
            if "session_date" in mem_props:
                props["session_date"] = str(rec.metadata.get("session_date") or "")
            self.store.upsert_vertex(mem_label.name, rec.id, props)
            report.vertices_written += 1

        # 2) Entity 顶点（跨记忆去重）+ MENTIONS 边
        entities: dict[str, dict] = {}
        for rec in memories:
            for ent in extraction.get(rec.id, []):
                key = _scoped_key(str(rec.metadata["user_id"]), ent["name"])
                if not key:
                    continue
                if key not in entities:
                    props = {"entity_id": key, "name": ent["name"], "type": ent.get("type", "other")}
                    if ent.get("date") and "date" in ent_props:
                        props["date"] = str(ent["date"])
                    self.store.upsert_vertex(ent_label.name, key, props)
                    entities[key] = props
                    report.vertices_written += 1
                self.store.upsert_edge("MENTIONS", mem_label.name, rec.id, ent_label.name, key)
                report.edges_written += 1

        report.duration_s = round(time.perf_counter() - t0, 3)
        report.label_counts = {
            mem_label.name: len(memories),
            ent_label.name: len(entities),
        }
        return report


def make_schema_s0_import():
    from schema_rsi.graph.schema import make_schema_s0

    return make_schema_s0()
