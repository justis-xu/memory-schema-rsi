"""实体多级概念化（AutoSchemaKG 式）：每个实体生成 3-5 个不同抽象层级的短语节点。

论文依据（AutoSchemaKG 消融）：概念化边（instance_of）是跨断开子图的语义桥，
是多跳 QA 的最大增益来源；查询常用抽象类别词（"公司"/"运动"/"工作变动"）而
记忆条目只有具体实例。与 S2 的固定词表 Concept 正交——这里是自由短语、按实体
生成、带记忆上下文。

构建产物（对接 variants 2-hop 检索模式）：
  V10Concept(concept_key pk, name, user_id)
  Memory -[V10_ABSTRACT]-> V10Concept   ← 记忆直接连到其全部实体的抽象概念
缓存：data/graph/concept_cache.json（{entity_key: [phrase, ...]}）
"""

from __future__ import annotations

import json
import logging

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.chat import ChatClient, make_chat_client

logger = logging.getLogger(__name__)

CONCEPT_SYSTEM = "You are a knowledge-graph conceptualization engine. Output valid JSON only."

CONCEPT_PROMPT = """For each entity below, generate abstraction phrases used to group it with related things.

Rules:
- 3 to 5 phrases per entity, each 1-2 words, lowercase
- DIFFERENT abstraction levels: specific category first, then broader domains (e.g. "DoorDash" -> ["food delivery", "company", "employer", "gig economy"])
- Phrases must NOT reuse words from the entity name
- Use the context snippets to pick apt levels; prefer phrases a person might use when asking about this topic generically

Entities:
{entities}

Output JSON mapping each entity key to its phrases:
{{"<key>": ["phrase1", "phrase2"]}}

JSON:"""


class EntityConceptualizer:
    def __init__(self, settings: Settings | None = None, client: ChatClient | None = None):
        self.settings = settings or get_settings()
        self.client = client or make_chat_client(self.settings)
        self.cache_path = self.settings.resolve_path("data/graph/concept_cache.json")
        self._cache: dict[str, list[str]] = {}
        try:
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._cache = {}

    def flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    def conceptualize(self, entities: list[dict]) -> dict[str, list[str]]:
        """entities: [{key, name, type, context}]（context = 提及该实体的记忆片段，可选）。

        返回 {entity_key: [phrase, ...]}；失败批容忍，缓存续跑。
        """
        import re as _re

        pending = [e for e in entities if e.get("key") and e["key"] not in self._cache]
        batch_size = 15
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            lines = []
            for e in batch:
                ctx = " | ".join((e.get("context") or [])[:2])[:220]
                lines.append(f"[{e['key']}] {e['name']} (type: {e.get('type', 'other')})"
                             + (f" — context: {ctx}" if ctx else ""))
            try:
                text, _ = self.client.complete(
                    system=CONCEPT_SYSTEM,
                    user=CONCEPT_PROMPT.replace("{entities}", "\n".join(lines)),
                    max_tokens=1200,
                )
                m = _re.search(r"\{.*\}", text, _re.S)
                if not m:
                    raise ValueError("no JSON object")
                parsed = json.loads(m.group(0))
                for e in batch:
                    v = parsed.get(e["key"])
                    if isinstance(v, list):
                        phrases = [str(p).strip().lower()[:40] for p in v if str(p).strip()][:5]
                        if phrases:
                            self._cache[e["key"]] = phrases
            except Exception as err:  # noqa: BLE001
                logger.warning("conceptualization batch failed: %s", str(err)[:120])
            if (i // batch_size) % 10 == 0:
                self.flush()
        self.flush()
        return {e["key"]: self._cache.get(e["key"], []) for e in entities}
