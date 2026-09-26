"""图变体注册表：每个变体在共享 Memory 顶点池上叠加自己的节点/边（命名空间隔离）。

设计（100 轮多图探索的基础设施）：
- 变体 = 顶点标签 + 边标签 + 检索跳转声明（hops）+ 幂等构建器
- 图不要求人类可读：聚类、共现对、月份链等全是机器抽象
- 构建幂等：完成标记 data/graph/variants/<id>.json，重复调用零成本
- S1（Entity/Event/Preference）与 S2（+Concept）视为基础变体，已建好

变体清单（wave 1，全部零 LLM 成本）：
  s1        基础结构化图（Entity/Event/Preference）
  s2        s1 + 抽象主题 Concept
  v4session 会话节点：同会话记忆聚合（上下文局部性）
  v5month   月份节点：YYYY-MM 时间片聚合（temporal 题型）
  v6pair    实体共现对节点：同一记忆内共同出现的实体对（multi-hop 联合上下文）
  v8cluster 嵌入 k-means 聚类节点：纯机器抽象主题（与 LLM 主题正交）
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from schema_rsi.config import Settings, get_settings
from schema_rsi.memory.base import MemoryRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 结构定义

@dataclass
class VariantSpec:
    vid: str
    description: str
    node_labels: list[str]
    hops: list[tuple[str, str, str]]  # (节点label, Memory出边, 节点主键属性)
    build: object | None = None  # callable(store, memories, settings) -> dict counts
    needs_embeddings: bool = False


def _base_s1_hops() -> list[tuple[str, str, str]]:
    return [
        ("Entity", "MENTIONS", "entity_id"),
        ("Event", "DESCRIBES_EVENT", "event_id"),
        ("Preference", "EXPRESSES_PREFERENCE", "pref_id"),
    ]


def _build_v4_session(store, memories: list[MemoryRecord], settings) -> dict:
    """会话节点：V4Session(session_key) + Memory-[V4_IN_SESSION]->Session。"""
    from schema_rsi.graph.schema import EdgeLabelSpec, GraphSchema, IndexSpec, PropertySpec, VertexLabelSpec

    schema = GraphSchema(
        name="v4session", version="v4",
        vertex_labels=[VertexLabelSpec(
            name="V4Session", primary_key="session_key",
            properties=[PropertySpec("session_key", "TEXT"), PropertySpec("session_id", "TEXT"),
                        PropertySpec("session_date", "TEXT"), PropertySpec("user_id", "TEXT")])],
        edge_labels=[EdgeLabelSpec(name="V4_IN_SESSION", source_labels=["Memory"],
                                   target_labels=["V4Session"])],
        indexes=[IndexSpec(name="v4SessionByKey", label="V4Session", field="session_key")],
    )
    store.create_schema(schema)
    n = 0
    for rec in memories:
        sid = str((rec.metadata or {}).get("session_id") or "")
        owner = str((rec.metadata or {}).get("user_id") or "")
        if not sid or not owner:
            continue
        key = f"{len(owner)}:{owner}:{sid}"
        store.upsert_vertex("V4Session", key, {"session_key": key, "session_id": sid,
                                               "session_date": str((rec.metadata or {}).get("session_date") or ""),
                                               "user_id": owner})
        store.upsert_edge("V4_IN_SESSION", "Memory", rec.id, "V4Session", key)
        n += 1
    return {"V4Session": len({(r.metadata or {}).get("user_id", "") + ":" + str((r.metadata or {}).get("session_id", ""))
                              for r in memories if (r.metadata or {}).get("session_id")}), "edges": n}


def _month_of(rec) -> str:
    iso = str((rec.metadata or {}).get("session_date") or "")
    return iso[:7] if len(iso) >= 7 else ""


def _build_v5_month(store, memories: list[MemoryRecord], settings) -> dict:
    """月份节点：V5Month(YYYY-MM) + Memory-[V5_IN_MONTH]->Month。"""
    from schema_rsi.graph.schema import EdgeLabelSpec, GraphSchema, IndexSpec, PropertySpec, VertexLabelSpec

    schema = GraphSchema(
        name="v5month", version="v5",
        vertex_labels=[VertexLabelSpec(
            name="V5Month", primary_key="month_key",
            properties=[PropertySpec("month_key", "TEXT"), PropertySpec("month", "TEXT"),
                        PropertySpec("user_id", "TEXT")])],
        edge_labels=[EdgeLabelSpec(name="V5_IN_MONTH", source_labels=["Memory"], target_labels=["V5Month"])],
        indexes=[IndexSpec(name="v5MonthByKey", label="V5Month", field="month_key")],
    )
    store.create_schema(schema)
    months: set[str] = set()
    n = 0
    for rec in memories:
        owner = str((rec.metadata or {}).get("user_id") or "")
        month = _month_of(rec)
        if not owner or not month:
            continue
        key = f"{len(owner)}:{owner}:{month}"
        months.add(key)
        store.upsert_vertex("V5Month", key, {"month_key": key, "month": month, "user_id": owner})
        store.upsert_edge("V5_IN_MONTH", "Memory", rec.id, "V5Month", key)
        n += 1
    return {"V5Month": len(months), "edges": n}


def _build_v6_pair(store, memories: list[MemoryRecord], settings) -> dict:
    """实体共现对：复用 S1 结构化缓存，同一记忆内实体两两组合 → V6Pair 节点。"""
    from schema_rsi.graph.schema import EdgeLabelSpec, GraphSchema, IndexSpec, PropertySpec, VertexLabelSpec

    cache_path = settings.resolve_path("data/graph/structured_cache.json")
    try:
        extraction = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("v6pair 需要 structured_cache.json（先跑过 S1 提取）")

    schema = GraphSchema(
        name="v6pair", version="v6",
        vertex_labels=[VertexLabelSpec(
            name="V6Pair", primary_key="pair_key",
            properties=[PropertySpec("pair_key", "TEXT"), PropertySpec("name", "TEXT"),
                        PropertySpec("user_id", "TEXT")])],
        edge_labels=[EdgeLabelSpec(name="V6_CO_MENTIONS", source_labels=["Memory"], target_labels=["V6Pair"])],
        indexes=[IndexSpec(name="v6PairByKey", label="V6Pair", field="pair_key")],
    )
    store.create_schema(schema)
    pairs: set[str] = set()
    n = 0
    for rec in memories:
        owner = str((rec.metadata or {}).get("user_id") or "")
        names = sorted({str(e.get("name", "")).strip()
                        for e in (extraction.get(rec.id) or {}).get("entities", [])
                        if str(e.get("name", "")).strip()})
        for i in range(len(names)):
            for j in range(i + 1, min(i + 4, len(names))):  # 每记忆最多前 3 个组合，控爆度
                pname = f"{names[i]}|{names[j]}"
                key = f"{len(owner)}:{owner}:{pname.lower()}"
                pairs.add(key)
                store.upsert_vertex("V6Pair", key, {"pair_key": key, "name": pname, "user_id": owner})
                store.upsert_edge("V6_CO_MENTIONS", "Memory", rec.id, "V6Pair", key)
                n += 1
    return {"V6Pair": len(pairs), "edges": n}


def _embed_memories(memories: list[MemoryRecord], settings) -> dict[str, list[float]]:
    """批量嵌入记忆（DashScope compatible 模式），缓存路径可经 graph.memory_embeddings_cache 配置
    （默认 data/graph/memory_embeddings.json，英文轨道不变；中文轨道指 data/graph_zh/）。"""
    import numpy as np
    from openai import OpenAI

    cache_path = settings.resolve_path(
        (settings.raw.get("graph") or {}).get(
            "memory_embeddings_cache", "data/graph/memory_embeddings.json"
        )
    )
    cache: dict[str, list] = {}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    pending = [r for r in memories if r.id and r.id not in cache]
    if pending:
        client = OpenAI(base_url=settings.embedding.base_url, api_key=settings.embedding.api_key)
        for i in range(0, len(pending), 25):
            batch = pending[i : i + 25]
            resp = client.embeddings.create(model=settings.embedding.model,
                                            input=[r.content[:1500] for r in batch])
            for r, item in zip(batch, resp.data):
                cache[r.id] = [round(float(x), 5) for x in item.embedding]
            if (i // 25) % 10 == 0:
                logger.info("embedding %d/%d", i + len(batch), len(pending))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return cache


def _kmeans(X, k: int, iters: int = 25, seed: int = 7):
    import numpy as np

    rng = np.random.default_rng(seed)
    k = min(k, len(X))
    # k-means++ 初始化
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        probs = d2 / max(d2.sum(), 1e-9)
        centers.append(X[rng.choice(len(X), p=probs)])
    C = np.array(centers)
    assign = None
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new_assign = d2.argmin(1)
        if assign is not None and (new_assign == assign).all():
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                C[c] = X[mask].mean(0)
    return assign if assign is not None else np.zeros(len(X), dtype=int), k


def _build_v8_cluster(store, memories: list[MemoryRecord], settings) -> dict:
    """嵌入 k-means 聚类节点（每用户独立聚类，k 随记忆量自适应）——纯机器抽象。"""
    import numpy as np
    from schema_rsi.graph.schema import EdgeLabelSpec, GraphSchema, IndexSpec, PropertySpec, VertexLabelSpec

    emb = _embed_memories(memories, settings)
    by_user: dict[str, list[MemoryRecord]] = {}
    for rec in memories:
        if rec.id in emb:
            by_user.setdefault(str((rec.metadata or {}).get("user_id") or ""), []).append(rec)

    schema = GraphSchema(
        name="v8cluster", version="v8",
        vertex_labels=[VertexLabelSpec(
            name="V8Cluster", primary_key="cluster_key",
            properties=[PropertySpec("cluster_key", "TEXT"), PropertySpec("name", "TEXT"),
                        PropertySpec("user_id", "TEXT")])],
        edge_labels=[EdgeLabelSpec(name="V8_IN_CLUSTER", source_labels=["Memory"], target_labels=["V8Cluster"])],
        indexes=[IndexSpec(name="v8ClusterByKey", label="V8Cluster", field="cluster_key")],
    )
    store.create_schema(schema)
    clusters = 0
    edges = 0
    for owner, recs in by_user.items():
        k = max(4, min(12, len(recs) // 25))
        X = np.array([emb[r.id] for r in recs])
        assign, k = _kmeans(X, k)
        for c in range(k):
            key = f"{len(owner)}:{owner}:cl{c}"
            store.upsert_vertex("V8Cluster", key, {"cluster_key": key, "name": f"cluster{c}", "user_id": owner})
            clusters += 1
        for rec, c in zip(recs, assign):
            store.upsert_edge("V8_IN_CLUSTER", "Memory", rec.id, "V8Cluster", f"{len(owner)}:{owner}:cl{int(c)}")
            edges += 1
    return {"V8Cluster": clusters, "edges": edges}


def _build_v10_concept(store, memories: list[MemoryRecord], settings) -> dict:
    """AutoSchemaKG 式实体多级概念化：实体 → 3-5 个抽象短语 → V10Concept 节点。

    记忆直接连到其全部实体的抽象概念（Memory-[V10_ABSTRACT]->V10Concept），
    与固定词表的 S2 Concept 正交。需要一次 LLM 批量概念化（缓存续跑）。
    """
    import re

    from schema_rsi.graph.extractor_concepts import EntityConceptualizer
    from schema_rsi.graph.schema import EdgeLabelSpec, GraphSchema, IndexSpec, PropertySpec, VertexLabelSpec

    cache_path = settings.resolve_path("data/graph/structured_cache.json")
    try:
        extraction = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("v10concept 需要 structured_cache.json（先跑过 S1 提取）")

    # 收集实体（含提及上下文）与 记忆→实体 映射
    def _scoped(owner: str, name: str) -> str:
        return f"{len(owner)}:{owner}:{' '.join(str(name).lower().strip().split())}"

    ent_info: dict[str, dict] = {}
    mem_ents: dict[str, list[str]] = {}
    for rec in memories:
        owner = str((rec.metadata or {}).get("user_id") or "")
        if not owner:
            continue
        for ent in (extraction.get(rec.id) or {}).get("entities", []):
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            key = _scoped(owner, name)
            info = ent_info.setdefault(key, {"key": key, "name": name,
                                             "type": ent.get("type", "other"), "context": []})
            if len(info["context"]) < 2:
                info["context"].append(rec.content[:160])
            mem_ents.setdefault(rec.id, []).append(key)

    concepts_by_ent = EntityConceptualizer(settings).conceptualize(list(ent_info.values()))

    schema = GraphSchema(
        name="v10concept", version="v10",
        vertex_labels=[VertexLabelSpec(
            name="V10Concept", primary_key="concept_key",
            properties=[PropertySpec("concept_key", "TEXT"), PropertySpec("name", "TEXT"),
                        PropertySpec("user_id", "TEXT")])],
        edge_labels=[EdgeLabelSpec(name="V10_ABSTRACT", source_labels=["Memory"],
                                   target_labels=["V10Concept"])],
        indexes=[IndexSpec(name="v10ConceptByKey", label="V10Concept", field="concept_key")],
    )
    store.create_schema(schema)
    nodes: set[str] = set()
    edges = 0
    for rec in memories:
        owner = str((rec.metadata or {}).get("user_id") or "")
        if not owner:
            continue
        seen: set[str] = set()
        for ekey in mem_ents.get(rec.id, []):
            for phrase in concepts_by_ent.get(ekey, []):
                ckey = _scoped(owner, phrase)
                if not ckey or ckey in seen:
                    continue
                seen.add(ckey)
                nodes.add(ckey)
                store.upsert_vertex("V10Concept", ckey,
                                    {"concept_key": ckey, "name": phrase, "user_id": owner})
                store.upsert_edge("V10_ABSTRACT", "Memory", rec.id, "V10Concept", ckey)
                edges += 1
    return {"V10Concept": len(nodes), "edges": edges}


S1_SPEC = VariantSpec("s1", "基础结构化图", ["Entity", "Event", "Preference"], _base_s1_hops(), None)
S2_SPEC = VariantSpec("s2", "s1+抽象主题", S1_SPEC.node_labels + ["Concept"],
                      _base_s1_hops() + [("Concept", "HAS_TOPIC", "concept_id")], None)

VARIANTS: dict[str, VariantSpec] = {
    "s1": S1_SPEC,
    "s2": S2_SPEC,
    "v4session": VariantSpec("v4session", "会话聚合", ["V4Session"],
                             [("V4Session", "V4_IN_SESSION", "session_key")], _build_v4_session),
    "v5month": VariantSpec("v5month", "月份聚合", ["V5Month"],
                           [("V5Month", "V5_IN_MONTH", "month_key")], _build_v5_month),
    "v6pair": VariantSpec("v6pair", "实体共现对", ["V6Pair"],
                          [("V6Pair", "V6_CO_MENTIONS", "pair_key")], _build_v6_pair),
    "v8cluster": VariantSpec("v8cluster", "嵌入聚类", ["V8Cluster"],
                             [("V8Cluster", "V8_IN_CLUSTER", "cluster_key")], _build_v8_cluster, True),
    "v10concept": VariantSpec("v10concept", "实体多级概念化(AutoSchemaKG式)", ["V10Concept"],
                              [("V10Concept", "V10_ABSTRACT", "concept_key")], _build_v10_concept),
}


def hops_for(variant_ids: list[str]) -> list[tuple[str, str, str]]:
    """变体组合 → 去重后的跳转声明（按 node label 去重）。"""
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for vid in variant_ids:
        spec = VARIANTS.get(vid)
        if spec is None:
            raise KeyError(f"unknown graph variant: {vid}")
        for hop in spec.hops:
            if hop[0] not in seen:
                seen.add(hop[0])
                out.append(hop)
    return out


def ensure_variant(store, variant_id: str, memories: list[MemoryRecord], settings: Settings | None = None) -> dict:
    """幂等构建：完成标记存在则跳过；s1/s2 检查现有顶点数即可。"""
    import os

    settings = settings or get_settings()
    spec = VARIANTS.get(variant_id)
    if spec is None:
        raise KeyError(f"unknown graph variant: {variant_id}")
    # marker 目录可经 graph.variants_dir 配置（默认英文轨道 data/graph/variants 不变；
    # 中文轨道用独立 marker 目录，避免英文 marker 让中文构建被误跳过）
    marker_dir = settings.resolve_path(
        (settings.raw.get("graph") or {}).get("variants_dir", "data/graph/variants")
    )
    marker = marker_dir / f"{variant_id}.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    if variant_id in ("s1", "s2"):  # 基础图：以服务端顶点数为准
        counts = {lab: store.count_vertices(lab) for lab in spec.node_labels}
        if counts.get(spec.node_labels[0]):
            out = {"preexisting": counts}
        else:
            raise RuntimeError(f"{variant_id} 需要先用既有脚本建图（eval_s1_ab/build_concepts）")
    else:
        t0 = time.perf_counter()
        counts = spec.build(store, memories, settings)
        counts["duration_s"] = round(time.perf_counter() - t0, 1)
        out = {"built": counts}
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(out, ensure_ascii=False))
    logger.info("variant %s ready: %s", variant_id, out)
    return out
