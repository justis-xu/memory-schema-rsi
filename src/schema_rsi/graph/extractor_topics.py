"""抽象主题（Concept）批量提取：为每条记忆分配 1-3 个受控词表主题。

Concept 是跨实体、跨会话的抽象聚合点（career/health/travel...），
服务于节点中心聚合召回：一个问题命中任一主题下记忆时，该主题的
全部记忆都可被带出。词表固定、允许记忆不含主题词字面（抽象层）。

结果缓存 data/graph/topic_cache.json（{memory_id: [topic, ...]}），
与 StructuredExtractor 同风格：批内失败容忍、全空缓存拒绝复用。
"""

from __future__ import annotations

import json
import logging

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.chat import ChatClient, make_chat_client

logger = logging.getLogger(__name__)

TOPIC_VOCAB = [
    "career", "education", "family", "relationship", "friends", "health",
    "fitness", "food", "cooking", "hobby", "arts", "music", "reading",
    "games", "sports", "travel", "outdoors", "pets", "home", "moving",
    "shopping", "finance", "technology", "volunteering", "community",
    "religion", "identity", "celebration", "plans", "achievements",
    "emotions", "transport", "weather", "safety",
]

TOPIC_SYSTEM = "You are a memory tagging engine. Output valid JSON only."

TOPIC_PROMPT = """Assign 1-3 abstract topics to each memory below for retrieval grouping.

Topics MUST come from this fixed list:
{vocab}

Rules:
- Choose the MOST SPECIFIC applicable topics (a memory about a job promotion gets "career", not also "plans").
- Topics are abstract categories; they rarely appear verbatim in the memory. "Went kayaking on the lake" -> outdoors, fitness.
- 1 topic is enough when clear; use 2-3 only when genuinely multi-domain.
- Every memory gets at least 1 topic.

Memories:
{memories}

Output JSON mapping each memory_id to its topic list:
{{"<memory_id>": ["topic1", "topic2"]}}

JSON:"""


class TopicExtractor:
    def __init__(self, settings: Settings | None = None, client: ChatClient | None = None):
        self.settings = settings or get_settings()
        self.client = client or make_chat_client(self.settings)
        self.cache_path = self.settings.resolve_path(
            (self.settings.raw.get("graph") or {}).get(
                "topic_cache", "data/graph/topic_cache.json"
            )
        )
        self._cache: dict[str, list[str]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._cache = {k: [t for t in v if t in TOPIC_VOCAB] for k, v in raw.items()}
        except (OSError, ValueError):
            self._cache = {}
        if self._cache and not any(self._cache.values()):
            logger.warning("topic cache is entirely empty; retrying extraction")
            self._cache = {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def extract(self, records) -> dict[str, list[str]]:
        """返回 {memory_id: [topic, ...]}；批失败容忍，结果落盘缓存。"""
        pending = [r for r in records if r.id and r.id not in self._cache]
        batch_size = int((self.settings.raw.get("graph") or {}).get("extract_batch_size", 10))
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            memories_text = "\n".join(f"[{r.id}] {r.content}" for r in batch)
            try:
                text, _ = self.client.complete(
                    system=TOPIC_SYSTEM,
                    user=TOPIC_PROMPT.replace("{vocab}", ", ".join(TOPIC_VOCAB))
                    .replace("{memories}", memories_text),
                    max_tokens=800,
                )
                parsed = self._parse_json(text)
                if not parsed:
                    raise ValueError("topic extraction returned no valid JSON object")
                for r in batch:
                    topics = parsed.get(r.id)
                    if isinstance(topics, list):
                        clean = [str(t) for t in topics if str(t) in TOPIC_VOCAB][:3]
                        self._cache[r.id] = clean  # 空列表 = 无主题，构图时跳过
            except Exception as e:  # noqa: BLE001
                logger.warning("topic extraction batch failed: %s", str(e)[:120])
            self._save_cache()
        return {r.id: self._cache.get(r.id, []) for r in records if r.id}

    @staticmethod
    def _parse_json(text: str) -> dict:
        import re

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except ValueError:
            return {}
