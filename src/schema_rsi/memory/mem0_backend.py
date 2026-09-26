"""Mem0 OSS 封装。

要点：
- Vector Store 用 Chroma（纯本地文件，无服务进程），路径/集合名来自配置，可替换。
- Embedder / LLM 均为 provider=openai + openai_base_url → 任意 OpenAI-compatible 端点。
- 刻意【不】配置 graph_store：图实验层由本项目自己的 GraphStore/GraphBuilder 负责，
  Mem0 只作为 Memory Source（事实源）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from schema_rsi.config import Settings, get_settings
from schema_rsi.memory.base import MemoryRecord

logger = logging.getLogger(__name__)

# mem0 2.x 的 get_all 有 top_k 上限（默认 20），实验场景调大
_GET_ALL_LIMIT = 1000


def build_mem0_config(settings: Settings) -> dict:
    """由 Settings 生成 mem0 Memory.from_config 所需的配置 dict（集中在配置层，不散落代码）。"""
    mem0_cfg = settings.mem0
    vector_store_path = settings.resolve_path(
        mem0_cfg.get("vector_store_path", "data/vector_store/chroma")
    )
    history_db_path = settings.resolve_path(
        mem0_cfg.get("history_db_path", "data/vector_store/mem0_history.db")
    )
    Path(vector_store_path).mkdir(parents=True, exist_ok=True)

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": settings.llm.model,
                "api_key": settings.llm.api_key,
                "openai_base_url": settings.llm.base_url,
                "temperature": settings.llm.temperature,
                "max_tokens": 4000,
                # [schema-rsi] 源码级开关（third_party/mem0-src）：关闭推理模型深度思考
                "disable_thinking": True,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": settings.embedding.model,
                "api_key": settings.embedding.api_key,
                "openai_base_url": settings.embedding.base_url,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": mem0_cfg.get("collection_name", "schema_rsi_memories"),
                "path": str(vector_store_path),
            },
        },
        "history_db_path": str(history_db_path),
        # 高密度事实提取（可选，config 的 mem0.fact_extraction_prompt；留空则用 mem0
        # 默认 prompt）。vendored mem0 2.1.0 的插口是顶层 custom_instructions
        #（mem0/configs/base.py:54）：在 V3 additive 提取 user prompt 末尾追加
        # "## Custom Instructions" 段（mem0/configs/prompts.py:1044），对话由
        # "## New Messages" 段自动拼接。旧名 custom_fact_extraction_prompt 在
        # MemoryConfig 无对应字段，会被 pydantic 静默丢弃（extra=ignore）。
        # {messages} 占位符在 2.1.0 不做字符串替换，仅为兼容本函数旧约定保留字面量。
        **(
            {"custom_instructions": mem0_cfg["fact_extraction_prompt"]}
            if mem0_cfg.get("fact_extraction_prompt")
            else {}
        ),
        # 注意：没有 graph_store 配置 —— 不使用 Mem0 Graph Memory。
    }


def _as_items(result: Any) -> list[dict]:
    """兼容 mem0 不同版本的返回结构（{"results":[...]} 或裸 list）。"""
    if isinstance(result, dict):
        return result.get("results", []) or []
    if isinstance(result, list):
        return result
    return []


def _to_record(item: dict, default_user_id: str | None = None) -> MemoryRecord:
    metadata = dict(item.get("metadata") or {})
    # mem0 2.x 把 user_id/agent_id 作为一等字段返回，映射进 metadata 统一承载
    for field in ("user_id", "agent_id", "run_id"):
        if item.get(field) and field not in metadata:
            metadata[field] = item[field]
    if default_user_id and metadata.get("user_id") is None:
        metadata.setdefault("user_id", default_user_id)
    return MemoryRecord(
        id=str(item.get("id", "")),
        content=item.get("memory", "") or "",
        metadata=metadata,
        score=item.get("score"),
    )


class Mem0Backend:
    """MemoryBackend 的 Mem0 OSS 实现。"""

    def __init__(self, settings: Settings | None = None, config: dict | None = None):
        from mem0 import Memory  # 延迟导入，加快纯数据集脚本的启动

        self.settings = settings or get_settings()
        self.config = config or build_mem0_config(self.settings)
        self._memory = Memory.from_config(self.config)

    # ---- MemoryBackend 协议实现 ----

    def add_memory(
        self, user_id: str, messages: list[dict], metadata: dict | None = None
    ) -> list[MemoryRecord]:
        # mem0 2.x 禁止在 metadata 里携带身份字段（user_id/agent_id/run_id）
        meta = {k: v for k, v in (metadata or {}).items()
                if k not in ("user_id", "agent_id", "run_id")}
        result = self._memory.add(messages, user_id=user_id, metadata=meta)
        records = []
        for item in _as_items(result):
            rec = _to_record(item, default_user_id=user_id)
            rec.metadata.setdefault("user_id", user_id)
            rec.metadata["event"] = item.get("event")
            records.append(rec)
        return records

    def search_memory(
        self, query: str, user_id: str | None = None, top_k: int | None = None
    ) -> list[MemoryRecord]:
        limit = top_k or self.settings.top_k
        filters = {"user_id": user_id} if user_id else None
        result = self._memory.search(query, top_k=limit, filters=filters)
        return [_to_record(item, default_user_id=user_id) for item in _as_items(result)]

    def get_all_memories(self, user_id: str | None = None) -> list[MemoryRecord]:
        if not user_id:
            raise ValueError(
                "mem0 2.x 的 get_all 必须带 user_id/agent_id/run_id 过滤，请显式传入 user_id"
            )
        result = self._memory.get_all(filters={"user_id": user_id}, top_k=_GET_ALL_LIMIT)
        return [_to_record(item, default_user_id=user_id) for item in _as_items(result)]

    def reset_memories(self, user_id: str | None = None) -> None:
        if user_id is None:
            self._memory.reset()
            return
        try:
            self._memory.delete_all(user_id=user_id)
        except Exception:  # 老版本 mem0 无 delete_all：逐条删除兜底
            for rec in self.get_all_memories(user_id=user_id):
                try:
                    self._memory.delete(rec.id)
                except Exception:  # pragma: no cover
                    logger.warning("failed to delete memory %s", rec.id)

    @property
    def raw(self):
        """暴露底层 mem0.Memory（调试用，正常流程不要依赖）。"""
        return self._memory
