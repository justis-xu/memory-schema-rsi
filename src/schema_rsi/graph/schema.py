"""动态、通用的图 Schema 描述（数据结构层，不含任何具体图数据库细节）。

这是未来 SchemaRSI 的落点之一：Schema S0/S1/S2... 都用 GraphSchema 表达，
GraphBuilder.build(memories, schema) 按给定 schema 重建图。

注意：make_test_schema() 只服务于 Phase 0 的基础设施冒烟验证，
【不是】最终 Memory Graph Schema 设计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# HugeGraph 支持的属性类型（TEXT/INT/DOUBLE/BOOLEAN/DATE/...），此处沿用其词表
PG_TYPES = {"TEXT", "INT", "DOUBLE", "BOOLEAN", "DATE", "LONG", "FLOAT", "UUID", "BLOB"}


@dataclass
class PropertySpec:
    name: str
    pg_type: str = "TEXT"          # 见 PG_TYPES
    cardinality: str = "single"    # single | list | set


@dataclass
class VertexLabelSpec:
    name: str
    properties: list[PropertySpec] = field(default_factory=list)
    primary_key: str | None = None  # None => AUTOMATIC id 策略；否则 PRIMARY_KEY 策略
    id_strategy: str = "AUTOMATIC"  # AUTOMATIC | PRIMARY_KEY | CUSTOMIZE_STRING ...

    def property_names(self) -> list[str]:
        return [p.name for p in self.properties]

    def nullable_names(self) -> list[str]:
        pk = self.primary_key
        return [p.name for p in self.properties if p.name != pk]


@dataclass
class EdgeLabelSpec:
    name: str
    source_labels: list[str] = field(default_factory=list)   # 空 = 任意
    target_labels: list[str] = field(default_factory=list)
    properties: list[PropertySpec] = field(default_factory=list)

    def property_names(self) -> list[str]:
        return [p.name for p in self.properties]


@dataclass
class IndexSpec:
    name: str
    label: str                      # vertex/edge label 名
    field: str = ""                 # 属性名
    index_type: str = "SECONDARY"   # SECONDARY | RANGE | SEARCH | SHARD | UNIQUE


@dataclass
class GraphSchema:
    """一套完整图 schema：顶点类型 + 边类型 + 索引。"""

    name: str
    vertex_labels: list[VertexLabelSpec] = field(default_factory=list)
    edge_labels: list[EdgeLabelSpec] = field(default_factory=list)
    indexes: list[IndexSpec] = field(default_factory=list)
    version: str = "v0"

    def vertex_label(self, name: str) -> VertexLabelSpec | None:
        return next((v for v in self.vertex_labels if v.name == name), None)

    def edge_label(self, name: str) -> EdgeLabelSpec | None:
        return next((e for e in self.edge_labels if e.name == name), None)

    def primary_vertex_label(self) -> VertexLabelSpec | None:
        return self.vertex_labels[0] if self.vertex_labels else None


def make_test_schema() -> GraphSchema:
    """Phase 0 基础设施冒烟测试用的【测试 Schema】——仅验证图数据库能力。

    明确标注：这不是最终 Memory Graph Schema 设计。未来由 SchemaRSI 产生
    S0/S1/S2... 时会替换为真正的 Entity/Event/State 等类型体系。
    """
    return GraphSchema(
        name="memory_test_v0",
        version="v0",
        vertex_labels=[
            VertexLabelSpec(
                name="Memory",
                primary_key="memory_id",   # 业务主键：GraphStore 的 (label, key) 即用它 upsert/查询
                id_strategy="AUTOMATIC",
                properties=[
                    PropertySpec("memory_id", "TEXT"),
                    PropertySpec("content", "TEXT"),
                    PropertySpec("user_id", "TEXT"),
                    PropertySpec("source", "TEXT"),
                    PropertySpec("created_at", "TEXT"),
                ],
            )
        ],
        edge_labels=[
            EdgeLabelSpec(
                name="RELATED_TO",
                source_labels=["Memory"],
                target_labels=["Memory"],
                properties=[PropertySpec("reason", "TEXT")],
            )
        ],
        indexes=[
            IndexSpec(
                name="memoryById",
                index_type="SECONDARY",
                label="Memory",
                field="memory_id",
            ),
            IndexSpec(
                name="memoryByUserId",
                index_type="SECONDARY",
                label="Memory",
                field="user_id",
            ),
        ],
    )


def make_schema_s1() -> GraphSchema:
    """Schema S1 —— 在 S0 基础上增加结构化日期事件和偏好关系。

    S0→S1 的设计假设（此前 S0 A/B 存在跨用户召回和上下文预算混杂，
    31/127 的翻正数字已撤回，不能作为效果证据）：
    - temporal 类问题需要更可靠的日期结构
      → Event 节点带结构化日期（YYYY-MM），不再依赖 LLM 在自由文本里锚定日期
    - open-domain 是最弱类别（76%），涉及偏好/推理
      → Preference 节点 + HAS_PREFERENCE 边
    - multi-hop 需要跨记忆关联（"Jon 和 Gina 的共同点"）
      → Entity-Entity RELATIONSHIP 边

    新增顶点：
      Event(event_id pk, name, date, participants, description)
      Preference(pref_id pk, subject, pref_type, object)

    新增边：
      Memory -[DESCRIBES_EVENT]-> Event
      Memory -[EXPRESSES_PREFERENCE]-> Preference
      Entity -[PARTICIPATES_IN]-> Event
      Entity -[HAS_PREFERENCE]-> Preference
      Entity -[RELATED_TO]-> Entity
    """
    return GraphSchema(
        name="memory_graph_s1",
        version="s1",
        vertex_labels=[
            VertexLabelSpec(
                name="Memory",
                primary_key="memory_id",
                id_strategy="AUTOMATIC",
                properties=[
                    PropertySpec("memory_id", "TEXT"),
                    PropertySpec("content", "TEXT"),
                    PropertySpec("user_id", "TEXT"),
                    PropertySpec("session_id", "TEXT"),
                    PropertySpec("session_date", "TEXT"),
                    PropertySpec("source", "TEXT"),
                ],
            ),
            VertexLabelSpec(
                name="Entity",
                primary_key="entity_id",
                id_strategy="AUTOMATIC",
                properties=[
                    PropertySpec("entity_id", "TEXT"),
                    PropertySpec("name", "TEXT"),
                    PropertySpec("type", "TEXT"),
                    PropertySpec("date", "TEXT"),
                ],
            ),
            VertexLabelSpec(
                name="Event",
                primary_key="event_id",
                id_strategy="AUTOMATIC",
                properties=[
                    PropertySpec("event_id", "TEXT"),
                    PropertySpec("name", "TEXT"),
                    PropertySpec("date", "TEXT"),  # YYYY-MM or YYYY-MM-DD（结构化，非自由文本）
                    PropertySpec("participants", "TEXT"),
                    PropertySpec("description", "TEXT"),
                ],
            ),
            VertexLabelSpec(
                name="Preference",
                primary_key="pref_id",
                id_strategy="AUTOMATIC",
                properties=[
                    PropertySpec("pref_id", "TEXT"),
                    PropertySpec("subject", "TEXT"),    # 谁的偏好
                    PropertySpec("pref_type", "TEXT"),  # like / dislike / prefer / avoid
                    PropertySpec("object", "TEXT"),     # 偏好什么
                ],
            ),
        ],
        edge_labels=[
            EdgeLabelSpec(
                name="MENTIONS",
                source_labels=["Memory"],
                target_labels=["Entity"],
                properties=[],
            ),
            EdgeLabelSpec(
                name="DESCRIBES_EVENT",
                source_labels=["Memory"],
                target_labels=["Event"],
                properties=[],
            ),
            EdgeLabelSpec(
                name="EXPRESSES_PREFERENCE",
                source_labels=["Memory"],
                target_labels=["Preference"],
                properties=[],
            ),
            EdgeLabelSpec(
                name="PARTICIPATES_IN",
                source_labels=["Entity"],
                target_labels=["Event"],
                properties=[],
            ),
            EdgeLabelSpec(
                name="HAS_PREFERENCE",
                source_labels=["Entity"],
                target_labels=["Preference"],
                properties=[],
            ),
            EdgeLabelSpec(
                name="RELATED_TO",
                source_labels=["Entity"],
                target_labels=["Entity"],
                properties=[PropertySpec("relation", "TEXT")],  # knows/works_at/family_of/member_of
            ),
        ],
        indexes=[
            IndexSpec(name="s1MemById", label="Memory", field="memory_id"),
            IndexSpec(name="s1MemByUser", label="Memory", field="user_id"),
            IndexSpec(name="s1EntById", label="Entity", field="entity_id"),
            IndexSpec(name="s1EntByName", label="Entity", field="name"),
            IndexSpec(name="s1EvtById", label="Event", field="event_id"),
            IndexSpec(name="s1EvtByDate", label="Event", field="date"),
            IndexSpec(name="s1PrefById", label="Preference", field="pref_id"),
        ],
    )


def make_schema_s2() -> GraphSchema:
    """Schema S2 —— S1 + 抽象 Concept（主题）节点。

    设计动机（用户方向：图融入检索主流程，节点不要求人类可读）：
    Entity/Event 是"具体名词"聚合点；Concept 是跨实体、跨会话的抽象主题
    聚合点（career / health / travel ...）。检索时一个 Concept 节点一次
    带出该主题下的全部记忆，补足"问题问的是领域而非具体实体"的召回。

    新增（可在已有 S1 图上增量创建，不 reset）：
      Concept(concept_id pk, name, user_id)
      Memory -[HAS_TOPIC]-> Concept
    """
    s1 = make_schema_s1()
    return GraphSchema(
        name="memory_graph_s2",
        version="s2",
        vertex_labels=s1.vertex_labels + [
            VertexLabelSpec(
                name="Concept",
                primary_key="concept_id",
                id_strategy="AUTOMATIC",
                properties=[
                    PropertySpec("concept_id", "TEXT"),
                    PropertySpec("name", "TEXT"),
                    PropertySpec("user_id", "TEXT"),
                ],
            ),
        ],
        edge_labels=s1.edge_labels + [
            EdgeLabelSpec(
                name="HAS_TOPIC",
                source_labels=["Memory"],
                target_labels=["Concept"],
                properties=[],
            ),
        ],
        indexes=s1.indexes + [
            IndexSpec(name="s2ConceptById", label="Concept", field="concept_id"),
        ],
    )


def make_schema_s0() -> GraphSchema:
    """Schema S0 —— 第一个正式的 Memory Graph Schema（手写，非自动演化）。

    设计动机（来自无图基线的 badcase 归因，results/full_locomo_*.jsonl）：
    - 检索召回不足（错题 20%）：MENTIONS 边提供"命中记忆 → 实体 → 关联记忆"的一跳扩展
    - 作答综合错（错题 50%）：Entity 节点把同一实体的事实聚合到一处
    - 日期类错误：实体带 date（activity/event 类），聚合时可见时间结构

    顶点：
      Memory(memory_id pk, content, user_id, session_id, session_date, source)
      Entity(entity_id pk=规范名, name, type, date?)
        type ∈ {person, place, organization, activity, event, preference, object, other}
    边：
      Memory -[MENTIONS]-> Entity
    """
    return GraphSchema(
        name="memory_graph_s0",
        version="s0",
        vertex_labels=[
            VertexLabelSpec(
                name="Memory",
                primary_key="memory_id",
                id_strategy="AUTOMATIC",
                properties=[
                    PropertySpec("memory_id", "TEXT"),
                    PropertySpec("content", "TEXT"),
                    PropertySpec("user_id", "TEXT"),
                    PropertySpec("session_id", "TEXT"),
                    PropertySpec("session_date", "TEXT"),
                    PropertySpec("source", "TEXT"),
                ],
            ),
            VertexLabelSpec(
                name="Entity",
                primary_key="entity_id",
                id_strategy="AUTOMATIC",
                properties=[
                    PropertySpec("entity_id", "TEXT"),
                    PropertySpec("name", "TEXT"),
                    PropertySpec("type", "TEXT"),
                    PropertySpec("date", "TEXT"),
                ],
            ),
        ],
        edge_labels=[
            EdgeLabelSpec(
                name="MENTIONS",
                source_labels=["Memory"],
                target_labels=["Entity"],
                properties=[],
            ),
        ],
        indexes=[
            IndexSpec(name="s0MemoryById", label="Memory", field="memory_id"),
            IndexSpec(name="s0MemoryByUser", label="Memory", field="user_id"),
            IndexSpec(name="s0EntityById", label="Entity", field="entity_id"),
            IndexSpec(name="s0EntityByName", label="Entity", field="name"),
        ],
    )
