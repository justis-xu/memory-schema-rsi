"""记忆后端抽象。

Vector Store（当前为 Chroma）完全封装在 Mem0Backend 内部，未来可整体替换，
上层（GraphBuilder / EvaluationPipeline）只依赖本协议与 MemoryRecord。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MemoryRecord:
    """一条长期记忆的最小统一表示。

    metadata 尽量保留: user_id / session_id / benchmark / case_id / created_at 等。
    """

    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float | None = None


@runtime_checkable
class MemoryBackend(Protocol):
    def add_memory(
        self, user_id: str, messages: list[dict], metadata: dict | None = None
    ) -> list[MemoryRecord]:
        """写入一段对话（messages: [{"role","content"}, ...]），返回产生/更新的记忆。"""
        ...

    def search_memory(
        self, query: str, user_id: str | None = None, top_k: int | None = None
    ) -> list[MemoryRecord]:
        ...

    def get_all_memories(self, user_id: str | None = None) -> list[MemoryRecord]:
        ...

    def reset_memories(self, user_id: str | None = None) -> None:
        """user_id=None 时清空全部记忆。"""
        ...
