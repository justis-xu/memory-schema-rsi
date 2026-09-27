"""Retriever 层。

- Mem0Retriever: 向量检索（可选 rerank 重排）。
- GraphRetriever: V0 极简图检索 —— 按已命中记忆在图上的顶点取 1 跳邻居内容。
  这不是正式 Graph RAG 设计，仅验证"图链路可用于对照实验"。
"""

from __future__ import annotations

import logging
import time

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.rerank import RerankClient, RerankError
from schema_rsi.memory.base import MemoryBackend, MemoryRecord

logger = logging.getLogger(__name__)

# chroma 同进程同路径只能开一个 PersistentClient（mem0 已开），这里做进程级缓存
_CHROMA_CLIENTS: dict[str, object] = {}


def _shared_chroma(path: str):
    if path not in _CHROMA_CLIENTS:
        import chromadb

        _CHROMA_CLIENTS[path] = chromadb.PersistentClient(path=path)
    return _CHROMA_CLIENTS[path]


class Mem0Retriever:
    def __init__(self, backend: MemoryBackend, settings: Settings | None = None):
        self.backend = backend
        self.settings = settings or get_settings()
        self._reranker: RerankClient | None = None
        if self.settings.rerank_enabled:
            self._reranker = RerankClient(
                base_url=self.settings.rerank.base_url,
                api_key=self.settings.rerank.api_key,
                model=self.settings.rerank.model,
            )
        self.last_rerank_error: str | None = None
        self.use_atomic = True  # 原子池默认启用（轮次轴 atomic: false 可关）
        # 原子事实池（提取层重做）：检索时与 mem0 主池 RRF 合流
        self.atomic_col = None
        self._atomic_emb = None
        try:
            from openai import OpenAI

            # 复用 mem0 已开的 chroma 客户端（同进程重复 open 同路径会因 settings 冲突报错）
            client = getattr(getattr(self.backend, "_memory", None), "vector_store", None)
            client = getattr(client, "client", None)
            if client is None:
                path = str(self.settings.resolve_path(self.settings.raw["mem0"]["vector_store_path"]))
                client = _shared_chroma(path)
            cols = {c.name for c in client.list_collections()}
            if "schema_rsi_atomic" in cols:
                self.atomic_col = client.get_collection("schema_rsi_atomic")
                self._atomic_emb = OpenAI(base_url=self.settings.embedding.base_url,
                                          api_key=self.settings.embedding.api_key)
        except Exception as e:  # noqa: BLE001
            logger.warning("atomic pool unavailable: %s", str(e)[:120])
            self.atomic_col = None

    def _atomic_search(self, query: str, user_id: str | None, top_k: int) -> list[MemoryRecord]:
        assert self.atomic_col is not None and self._atomic_emb is not None
        try:
            qv = self._atomic_emb.embeddings.create(
                model=self.settings.embedding.model, input=[query[:1500]]
            ).data[0].embedding
            res = self.atomic_col.query(
                query_embeddings=[qv], n_results=top_k,
                where={"user_id": user_id} if user_id else None,
            )
            metas = res.get("metadatas") or [[]]
            return [
                MemoryRecord(id=i, content=d or "", metadata=dict(m or {}))
                for i, d, m in zip((res.get("ids") or [[]])[0],
                                   (res.get("documents") or [[]])[0], metas[0])
            ][:top_k]
        except Exception as e:  # noqa: BLE001
            logger.warning("atomic search failed: %s", str(e)[:100])
            return []

    @staticmethod
    def _rrf_merge(a: list[MemoryRecord], b: list[MemoryRecord], top_n: int,
                   k: int = 60, b_weight: float = 1.25) -> list[MemoryRecord]:
        """RRF 合流两路召回（原子池略加权：原子事实更可能精确命中）。"""
        scores: dict[str, float] = {}
        recs: dict[str, MemoryRecord] = {}
        for rank, r in enumerate(a):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank + 1)
            recs[r.id] = r
        for rank, r in enumerate(b):
            scores[r.id] = scores.get(r.id, 0.0) + b_weight / (k + rank + 1)
            recs.setdefault(r.id, r)
        ranked = sorted(scores, key=lambda i: -scores[i])
        return [recs[i] for i in ranked[:top_n]]

    def rerank_pool(self, query: str, records: list[MemoryRecord], top_n: int) -> list[MemoryRecord] | None:
        """对一个候选池统一重排（融合模式共用选择器）。失败返回 None（调用方降级）。

        rerank 关闭时用零 API 成本的词面重叠兜底排序——否则池内向量结果永远
        排在图候选前面，融合模式退化为纯向量。
        """
        if len(records) <= top_n:
            return list(records[:top_n])
        if self._reranker is None:
            return self._lexical_rank(query, records, top_n)
        try:
            scored = self._reranker.rerank(query, [r.content for r in records], top_n=top_n)
            return [records[s["index"]] for s in scored if 0 <= s["index"] < len(records)][:top_n]
        except RerankError as e:
            self.last_rerank_error = str(e)
            logger.warning("pool rerank failed: %s", e)
            return None

    @staticmethod
    def _lexical_rank(query: str, records: list[MemoryRecord], top_n: int) -> list[MemoryRecord]:
        """词面重叠排序（rerank 不可用时的兜底，零 API 成本）。"""
        import re as _re

        stop = {"the", "a", "an", "in", "on", "at", "of", "to", "and", "or", "is",
                "was", "did", "do", "for", "with", "what", "which", "when", "where",
                "how", "who", "his", "her", "their", "they", "she", "he", "it"}
        q_tokens = {t for t in _re.findall(r"[a-z0-9]+", (query or "").lower())
                    if len(t) > 2 and t not in stop}
        if not q_tokens:
            return list(records[:top_n])

        def score(rec: MemoryRecord) -> float:
            c = set(_re.findall(r"[a-z0-9]+", (rec.content or "").lower()))
            return len(q_tokens & c) / len(q_tokens)

        ranked = sorted(records, key=score, reverse=True)
        return ranked[:top_n]

    def retrieve(self, query: str, user_id: str | None = None, top_k: int | None = None) -> list[MemoryRecord]:
        top_k = top_k or self.settings.top_k
        fetch_n = top_k * 3 if self._reranker else top_k  # 粗排多取一些供 rerank 筛选
        records = self.backend.search_memory(query, user_id=user_id, top_k=fetch_n)
        # 提取层重做：mem0 主池 + 原子事实池 RRF 合流（原子池略加权）
        if (getattr(self, "use_atomic", True) and self.atomic_col is not None
                and self.atomic_col.count()):
            atomic = self._atomic_search(query, user_id=user_id, top_k=fetch_n)
            if atomic:
                records = self._rrf_merge(records, atomic, top_n=len(records) + len(atomic))

        if self._reranker and len(records) > 1:
            t0 = time.perf_counter()
            try:
                scored = self._reranker.rerank(
                    query, [r.content for r in records], top_n=top_k
                )
                # scored 的 index 指向 documents 列表（即 records）的下标
                records = [
                    records[s["index"]] for s in scored if 0 <= s["index"] < len(records)
                ]
                records = records[:top_k]
                logger.debug("rerank done in %.3fs via %s", time.perf_counter() - t0, self._reranker.last_endpoint)
            except RerankError as e:
                # rerank 失败降级为原始向量序（记录原因，不阻断评测）
                self.last_rerank_error = str(e)
                logger.warning("rerank failed, fallback to vector order: %s", e)
                records = records[:top_k]
        return records[:top_k]


class GraphRetriever:
    """V0：命中记忆 → 图顶点 → 1 跳邻居内容。

    Schema S0（Memory+Entity+MENTIONS）下自动走双跳扩展：
    命中 Memory → MENTIONS → Entity → MENTIONS → 关联 Memory
    （这正是"补向量检索召回不足"的机制）。
    """

    def __init__(self, store, settings: Settings | None = None):
        self.store = store
        self.settings = settings or get_settings()

    @staticmethod
    def _same_owner(anchor: MemoryRecord, properties: dict) -> bool:
        """旧图也可能有跨用户共享节点；只放行可核对归属的记忆。"""
        owner = (anchor.metadata or {}).get("user_id")
        return bool(owner) and properties.get("user_id") == owner

    # 节点类型 → (节点label, Memory出边, 节点主键属性)
    _NODE_HOPS = (
        ("Entity", "MENTIONS", "entity_id"),
        ("Event", "DESCRIBES_EVENT", "event_id"),
        ("Preference", "EXPRESSES_PREFERENCE", "pref_id"),
        ("Concept", "HAS_TOPIC", "concept_id"),
    )

    def retrieve_fused(
        self,
        query_records: list[MemoryRecord],
        *,
        exclude_ids: set[str] | None = None,
        per_node_limit: int = 15,
        max_candidates: int = 30,
        hops: list[tuple[str, str, str]] | None = None,
    ) -> list[dict]:
        """节点中心聚合召回（融合模式）：种子记忆 → 关联节点 → 该节点下全部记忆。

        与 bypass 模式的区别：聚合目标是"节点下的所有记忆"（单节点上限
        per_node_limit），候选不按槽位截断；support（拉回该记忆的不同节点数）
        是融合前的图侧排序信号，最终由统一 rerank 与向量结果同池竞争。
        hops 可传变体组合（variants.hops_for），None 时自动探测已有节点类型。
        """
        exclude = set(exclude_ids or ())
        hop_list = hops or self._NODE_HOPS
        try:
            available = set(self.store._key_props)  # noqa: SLF001
        except Exception:
            available = set()

        agg: dict[str, dict] = {}
        order: list[str] = []
        for rec in query_records:
            for label, mem_edge, key_prop in hop_list:
                if label not in available:
                    continue
                nodes = self.store.get_neighbors(
                    "Memory", rec.id, direction="OUT", edge_labels=[mem_edge], limit=6
                )
                for node in nodes:
                    if node.get("label") != label:
                        continue
                    node_key = node["properties"].get(key_prop)
                    node_name = node["properties"].get("name") or node_key
                    linked = self.store.get_neighbors(
                        label, node_key, direction="IN",
                        edge_labels=[mem_edge], limit=per_node_limit,
                    )
                    for m in linked:
                        if m.get("label") != "Memory":
                            continue
                        mid = m["properties"].get("memory_id")
                        if (not mid or mid in exclude
                                or not self._same_owner(rec, m.get("properties") or {})):
                            continue
                        if mid not in agg:
                            agg[mid] = {
                                "id": mid,
                                "content": m["properties"].get("content", ""),
                                "user_id": m["properties"].get("user_id"),
                                "session_id": m["properties"].get("session_id"),
                                "session_date": m["properties"].get("session_date"),
                                "via": [],
                                "support": 0,
                                "anchor_memory_id": rec.id,
                            }
                            order.append(mid)
                        tag = f"{label.lower()}:{node_name}"
                        if tag not in agg[mid]["via"]:
                            agg[mid]["via"].append(tag)
                            agg[mid]["support"] += 1
        # support 降序（图侧信号），同 support 保持首次发现顺序
        ranked = sorted(order, key=lambda mid: -agg[mid]["support"])
        return [agg[mid] for mid in ranked[:max_candidates]]

    def retrieve(
        self,
        query_records: list[MemoryRecord],
        *,
        vertex_label: str = "Memory",
        edge_labels: list[str] | None = None,
        max_total: int = 15,
    ) -> list[dict]:
        # Schema 检测：S1（有 Event/Preference）> S0（有 Entity）> V0
        try:
            s1_mode = self.store._key_props.get("Event") is not None
            s0_mode = self.store._key_props.get("Entity") is not None
        except Exception:
            s1_mode = s0_mode = False

        if s1_mode:
            return self._retrieve_s1(query_records, max_total=max_total)
        if s0_mode:
            return self._retrieve_s0(query_records, max_total=max_total)

        out: list[dict] = []
        seen: set[str] = set()
        for rec in query_records:
            neighbors = self.store.get_neighbors(
                vertex_label, rec.id, direction="BOTH", edge_labels=edge_labels, limit=5
            )
            for n in neighbors:
                nid = str(n.get("id"))
                if (nid in seen or n.get("label") != vertex_label
                        or not self._same_owner(rec, n.get("properties") or {})):
                    continue
                seen.add(nid)
                out.append(
                    {
                        "id": n.get("properties", {}).get("memory_id", nid),
                        "content": n.get("properties", {}).get("content", ""),
                        "user_id": n.get("properties", {}).get("user_id"),
                        "session_id": n.get("properties", {}).get("session_id"),
                        "session_date": n.get("properties", {}).get("session_date"),
                        "via": n.get("via_edge"),
                        "anchor_memory_id": rec.id,
                    }
                )
                if len(out) >= max_total:
                    return out
        return out

    def _retrieve_s0(self, query_records: list[MemoryRecord], max_total: int) -> list[dict]:
        """S0 双跳：命中 Memory --MENTIONS--> Entity --MENTIONS--> 关联 Memory。"""
        out: list[dict] = []
        seen: set[str] = set()
        anchor_ids = {r.id for r in query_records}
        for rec in query_records:
            entities = self.store.get_neighbors(
                "Memory", rec.id, direction="OUT", edge_labels=["MENTIONS"], limit=8
            )
            for ent in entities:
                if ent.get("label") != "Entity":
                    continue
                linked = self.store.get_neighbors(
                    "Entity", ent["properties"].get("entity_id"), direction="IN",
                    edge_labels=["MENTIONS"], limit=8,
                )
                for m in linked:
                    if (m.get("label") != "Memory"
                            or not self._same_owner(rec, m.get("properties") or {})):
                        continue
                    mid = m["properties"].get("memory_id")
                    if mid in seen or mid in anchor_ids:
                        continue
                    seen.add(mid)
                    out.append(
                        {
                            "id": mid,
                            "content": m["properties"].get("content", ""),
                            "user_id": m["properties"].get("user_id"),
                            "session_id": m["properties"].get("session_id"),
                            "session_date": m["properties"].get("session_date"),
                            "via": f"entity:{ent['properties'].get('name')}",
                            "anchor_memory_id": rec.id,
                        }
                    )
                    if len(out) >= max_total:
                        return out
        return out

    def _retrieve_s1(self, query_records: list[MemoryRecord], max_total: int) -> list[dict]:
        """S1 多跳：实体跳 + 事件跳 + 偏好跳 + 关系跳，合并去重。"""
        out: list[dict] = []
        seen: set[str] = set()
        anchor_ids = {r.id for r in query_records}

        def _add(mid, content, via, anchor: MemoryRecord, properties: dict):
            if (mid and mid not in seen and mid not in anchor_ids and content
                    and self._same_owner(anchor, properties)):
                seen.add(mid)
                out.append({"id": mid, "content": content,
                            "user_id": properties.get("user_id"),
                            "session_id": properties.get("session_id"),
                            "session_date": properties.get("session_date"),
                            "via": via, "anchor_memory_id": anchor.id})
                return len(out) >= max_total
            return False

        for rec in query_records:
            # 1) Entity-hop（同 S0）
            entities = self.store.get_neighbors(
                "Memory", rec.id, direction="OUT", edge_labels=["MENTIONS"], limit=6
            )
            for ent in entities:
                if ent.get("label") != "Entity":
                    continue
                ent_name = ent["properties"].get("name", "?")
                linked = self.store.get_neighbors(
                    "Entity", ent["properties"].get("entity_id"), direction="IN",
                    edge_labels=["MENTIONS"], limit=6,
                )
                for m in linked:
                    if m.get("label") != "Memory":
                        continue
                    if _add(m["properties"].get("memory_id"),
                            m["properties"].get("content", ""),
                            f"entity:{ent_name}", rec, m["properties"]):
                        return out

            # 2) Event-hop（同一事件的其他记忆）
            events = self.store.get_neighbors(
                "Memory", rec.id, direction="OUT", edge_labels=["DESCRIBES_EVENT"], limit=5
            )
            for evt in events:
                if evt.get("label") != "Event":
                    continue
                evt_name = evt["properties"].get("name", "?")
                linked = self.store.get_neighbors(
                    "Event", evt["properties"].get("event_id"), direction="IN",
                    edge_labels=["DESCRIBES_EVENT"], limit=6,
                )
                for m in linked:
                    if m.get("label") != "Memory":
                        continue
                    if _add(m["properties"].get("memory_id"),
                            m["properties"].get("content", ""),
                            f"event:{evt_name}", rec, m["properties"]):
                        return out

            # 3) Preference-hop（相关偏好话题）
            prefs = self.store.get_neighbors(
                "Memory", rec.id, direction="OUT", edge_labels=["EXPRESSES_PREFERENCE"], limit=4
            )
            for pref in prefs:
                if pref.get("label") != "Preference":
                    continue
                obj = pref["properties"].get("object", "?")
                linked = self.store.get_neighbors(
                    "Preference", pref["properties"].get("pref_id"), direction="IN",
                    edge_labels=["EXPRESSES_PREFERENCE"], limit=5,
                )
                for m in linked:
                    if m.get("label") != "Memory":
                        continue
                    if _add(m["properties"].get("memory_id"),
                            m["properties"].get("content", ""),
                            f"pref:{obj}", rec, m["properties"]):
                        return out

        return out
