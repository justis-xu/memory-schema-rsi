"""结构化提取器：从记忆批量抽取实体 + 事件 + 偏好 + 关系（Schema S1 前置）。

一次 LLM 调用同时抽四类结构，输出 JSON；结果缓存（memory_id -> StructuredExtraction）。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.chat import ChatClient, make_chat_client

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = "You are a structured information extraction engine. Output valid JSON only."

EXTRACTION_PROMPT = """Extract structured information from each memory below.

Extract these 4 types:

1. **entities**: People, places, organizations, activities, notable objects
   - types: person, place, organization, activity, object, other
   - canonical name (e.g. "DoorDash" not "door dash")

2. **events**: Life events, milestones, achievements, appointments
   - Each event MUST have a date if determinable (YYYY-MM or YYYY-MM-DD)
   - Include participants as comma-separated names
   - Example: {"name": "lost job at DoorDash", "date": "2023-01", "participants": "Gina"}

3. **preferences**: Likes, dislikes, preferences, aversions
   - pref_type: like, dislike, prefer, avoid
   - subject: whose preference (person name or "user")
   - object: what they like/dislike
   - Example: {"subject": "Jon", "pref_type": "like", "object": "Marley flooring"}

4. **relationships**: Entity-to-entity relations
   - relation: knows, works_at, family_of, member_of, lives_in, owns
   - from_entity and to_entity are canonical entity names
   - Example: {"from_entity": "Jon", "relation": "lives_in", "to_entity": "downtown"}

Rules:
- 1-5 entities, 0-2 events, 0-2 preferences, 0-2 relationships per memory
- Use consistent entity naming across memories (same entity = same name)
- Events without a determinable date: omit the date field
- Skip generic words (user, memory, friend) unless part of a specific name

Memories:
{memories}

Output JSON for each memory_id:
{"<memory_id>": {"entities": [{"name":"...","type":"..."}], "events": [{"name":"...","date":"YYYY-MM","participants":"..."}], "preferences": [{"subject":"...","pref_type":"...","object":"..."}], "relationships": [{"from_entity":"...","relation":"...","to_entity":"..."}]}}

JSON:"""


class StructuredExtractor:
    """S1 级提取：实体 + 事件 + 偏好 + 关系，一次调用。"""

    def __init__(self, settings: Settings | None = None, client: ChatClient | None = None):
        self.settings = settings or get_settings()
        self.client = client or make_chat_client(self.settings)
        self.cache_path = self.settings.resolve_path(
            (self.settings.raw.get("graph") or {}).get(
                "structured_cache", "data/graph/structured_cache.json"
            )
        )
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._cache = {}
        # 旧版提示词格式化失败后把整批空结构写进了缓存；不能当成功结果复用。
        kinds = ("entities", "events", "preferences", "relationships")
        if self._cache and not any(
            any(item.get(kind) for kind in kinds)
            for item in self._cache.values() if isinstance(item, dict)
        ):
            logger.warning("structured extraction cache is entirely empty; retrying extraction")
            self._cache = {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def extract(self, records) -> dict[str, dict]:
        """返回 {memory_id: {"entities":[], "events":[], "preferences":[], "relationships":[]}}"""
        pending = [r for r in records if r.id and r.id not in self._cache]
        batch_size = int((self.settings.raw.get("graph") or {}).get("extract_batch_size", 10))
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            memories_text = "\n".join(f"[{r.id}] {r.content}" for r in batch)
            try:
                text, _ = self.client.complete(
                    system=EXTRACTION_SYSTEM,
                    user=EXTRACTION_PROMPT.replace("{memories}", memories_text),
                    max_tokens=2000,
                )
                parsed = self._parse_json(text)
                if not parsed:
                    raise ValueError("structured extraction returned no valid JSON object")
                for r in batch:
                    raw = parsed.get(r.id)
                    if isinstance(raw, dict):
                        self._cache[r.id] = self._normalize(raw)
            except Exception as e:
                logger.warning("structured extraction batch failed: %s", str(e)[:120])
            self._save_cache()
        return {r.id: self._cache.get(r.id, self._normalize({})) for r in records if r.id}

    @staticmethod
    def _normalize(raw: dict) -> dict:
        """清洗 LLM 输出的结构（去空、规范类型）。"""
        def clean_list(items, *fields):
            out = []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                d = {}
                for f in fields:
                    v = item.get(f)
                    if v is not None and str(v).strip():
                        d[f] = str(v).strip()
                if d:
                    out.append(d)
            return out

        return {
            "entities": clean_list(raw.get("entities"), "name", "type", "date"),
            "events": clean_list(raw.get("events"), "name", "date", "participants"),
            "preferences": clean_list(raw.get("preferences"), "subject", "pref_type", "object"),
            "relationships": clean_list(raw.get("relationships"), "from_entity", "relation", "to_entity"),
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
