"""直连 chroma 向量后端（BigModel 嵌入，绕过 mem0；评测只读检索）。

背景：百炼密钥失效/欠费期间，向量层切换到 BigModel embedding-3（LLM 同账号）。
旧的 mem0/chroma 库不动（保留历史可比性），新库在 data/chroma_bm/，由
scripts/rebuild_vector_bm.py 从旧库导出记忆后用 BigModel 嵌入重建。

与 Mem0Backend 同接口（search_memory / get_all_memories），id 与旧库一致，
保证图（Memory 顶点主键 = memory id）的融合检索仍然对得上。
"""

from __future__ import annotations

import logging

from schema_rsi.config import Settings, get_settings
from schema_rsi.memory.base import MemoryRecord

logger = logging.getLogger(__name__)


class ChromaDirectBackend:
    def __init__(self, settings: Settings | None = None):
        import chromadb
        from openai import OpenAI

        self.settings = settings or get_settings()
        self._col = chromadb.PersistentClient(
            path=self.settings.resolve_path("data/chroma_bm")
        ).get_or_create_collection("schema_rsi_bm")
        self._client = OpenAI(
            base_url=self.settings.embedding.base_url,
            api_key=self.settings.embedding.api_key,
        )
        self._model = self.settings.embedding.model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 25):
            resp = self._client.embeddings.create(
                model=self._model, input=[t[:2000] for t in texts[i : i + 25]]
            )
            out.extend(d.embedding for d in resp.data)
        return out

    # ---- MemoryBackend 协议（评测用到的子集） ----

    def search_memory(self, query: str, user_id: str | None = None, top_k: int = 10) -> list[MemoryRecord]:
        qv = self._embed([query])[0]
        res = self._col.query(
            query_embeddings=[qv],
            n_results=max(1, min(top_k, self._col.count())),
            where={"user_id": user_id} if user_id else None,
            include=["metadatas", "documents", "distances"],
        )
        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        return [
            MemoryRecord(id=i, content=d or "", metadata=dict(m or {}))
            for i, d, m in zip(ids[0], docs[0], metas[0])
        ][:top_k]

    def get_all_memories(self, user_id: str | None = None) -> list[MemoryRecord]:
        res = self._col.get(
            where={"user_id": user_id} if user_id else None,
            include=["metadatas", "documents"],
        )
        return [
            MemoryRecord(id=i, content=d or "", metadata=dict(m or {}))
            for i, d, m in zip(res.get("ids") or [], res.get("documents") or [], res.get("metadatas") or [])
        ]

    def add_memory(self, *args, **kwargs):  # pragma: no cover - 重建脚本直插，不走这里
        raise NotImplementedError("ChromaDirectBackend 是只读评测后端；写入用 rebuild_vector_bm.py")

    def reset_memories(self, user_id: str | None = None) -> None:  # pragma: no cover
        raise NotImplementedError("只读后端不支持 reset")
