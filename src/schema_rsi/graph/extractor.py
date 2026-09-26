"""实体抽取器：从已入库的记忆批量抽取实体（Schema S0 的建图前置）。

- 批量 LLM 调用（默认 10 条/批，JSON 输出），比逐条省 ~10 倍调用
- 结果缓存到 data/graph/entities_cache.json（memory_id -> [{name,type,date?}]），
  重复建图不重复付费（Memory=事实源，抽取结果也是派生物、可缓存）
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.chat import ChatClient, make_chat_client

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = "You are an information extraction engine. Output JSON only."

EXTRACTION_PROMPT = """Extract entities from each memory below.

Entity types: person, place, organization, activity, event, preference, object, other.
Rules:
- Include people's names, pets, places, brands, activities, events, preferences, and notable objects
- For activities/events with a date, include the date (YYYY-MM-DD or YYYY-MM if only month is known)
- Normalize obvious variants to a canonical name (e.g. "DoorDash" not "door dash")
- Skip generic words (user, memory, friend) unless part of a specific name
- 1-5 entities per memory; empty list if none

Memories:
{memories}

Output JSON: {{"<memory_id>": [{{"name": "...", "type": "...", "date": "YYYY-MM or null"}}]}}
JSON:"""


class EntityExtractor:
    def __init__(self, settings: Settings | None = None, client: ChatClient | None = None):
        self.settings = settings or get_settings()
        self.client = client or make_chat_client(self.settings)
        self.cache_path = self.settings.resolve_path(
            (self.settings.raw.get("graph") or {}).get("entities_cache", "data/graph/entities_cache.json")
        )
        self._cache: dict[str, list[dict]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._cache = {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def extract(self, records) -> dict[str, list[dict]]:
        """对 MemoryRecord 列表抽取实体，返回 {memory_id: [{name,type,date?}]}。

        已缓存的跳过；新抽取的分批调用并立即写缓存（断点安全）。
        """
        pending = [r for r in records if r.id and r.id not in self._cache]
        batch_size = int((self.settings.raw.get("graph") or {}).get("extract_batch_size", 10))
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            memories_text = "\n".join(f"[{r.id}] {r.content}" for r in batch)
            try:
                text, _ = self.client.complete(
                    system=EXTRACTION_SYSTEM,
                    user=EXTRACTION_PROMPT.format(memories=memories_text),
                    max_tokens=1500,
                )
                parsed = self._parse_json(text)
                for r in batch:
                    ents = parsed.get(r.id, [])
                    self._cache[r.id] = [
                        {
                            "name": str(e.get("name", "")).strip(),
                            "type": str(e.get("type", "other")).strip().lower(),
                            **({"date": str(e["date"])} if e.get("date") else {}),
                        }
                        for e in ents
                        if isinstance(e, dict) and e.get("name")
                    ]
            except Exception as e:  # 单批失败不阻断（记空并继续）
                logger.warning("entity extraction batch failed: %s", str(e)[:120])
                for r in batch:
                    self._cache.setdefault(r.id, [])
            self._save_cache()
        return {r.id: self._cache.get(r.id, []) for r in records if r.id}

    @staticmethod
    def _parse_json(text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except ValueError:
            return {}
