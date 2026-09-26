"""原子事实提取（提取层重做）：每个会话提取"一事实一条"的原子记忆。

动机（50 轮终审的归因）：single-hop 精准率 76.7% 封顶、列表题缺项，根因都是
mem0 默认提取的"合并摘要"丢细节——一条记忆塞多个事实、人名数字被省略、
列表只存前几项。原子化提取直接针对：
- 每个原子事实独立成条（subject + 具体事实 + 日期），禁止单条内合并
- 数字/名称/计数必须逐字保留（"three children" 不许写成 "children"）
- 列表逐项拆开（每项一条）
- 日期来自会话锚定 + 文本内显式日期

产物为独立向量池（schema_rsi_atomic 集合），与 mem0 原池并存，检索时合流——
不污染历史基线臂。缓存 data/extraction/atomic_cache.json（按 session）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.chat import ChatClient, make_chat_client

logger = logging.getLogger(__name__)

ATOMIC_SYSTEM = "You are a precise fact extraction engine. Output valid JSON only."

ATOMIC_PROMPT = """Extract EVERY atomic fact from this conversation session. One fact per item, never merge.

Session date: {session_date}
Speakers: {speakers}

Rules:
- ATOMIC: each item = exactly ONE fact about ONE subject. "Jon works at DoorDash and hates it" -> two items.
- PRESERVE LITERALS: names, numbers, counts, ages, dates verbatim ("three kids", "Prius", "January 2023").
- LISTS: one item per element (every book/game/child/place mentioned separately).
- DATES: use the explicit date in the text if any; otherwise "as of {session_date}" (the session date). Format: (YYYY-MM-DD) or (YYYY-MM).
- ATTRIBUTION: start with who the fact is about. In two-person chats, facts about either speaker count; note the source if the other speaker reported it.
- INCLUDE: events (with dates), states (jobs, homes, relationships), preferences, possessions, plans, health, milestones, opinions with specifics.
- EXCLUDE: small talk, greetings, meta-talk about the conversation itself.
- Output 5-25 items for a typical session. Format each as a short declarative sentence with the date tag at the end.

Session transcript:
{transcript}

Output JSON array of strings:
["...", "..."]

JSON:"""


class AtomicExtractor:
    def __init__(self, settings: Settings | None = None, client: ChatClient | None = None):
        self.settings = settings or get_settings()
        self.client = client or make_chat_client(self.settings)
        self.cache_path = Path(self.settings.resolve_path("data/extraction/atomic_cache.json"))
        self._cache: dict[str, list[str]] = {}
        try:
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._cache = {}

    def _flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    def extract_session(self, session_id: str, turns: list[dict], session_date: str,
                        speakers: str) -> list[str]:
        """turns: [{role/speaker, content}]。缓存按 session_id，失败容忍返回 []。"""
        if session_id in self._cache:
            return self._cache[session_id]
        transcript = "\n".join(
            f"{(t.get('speaker') or t.get('role') or '?')}: {t.get('content') or t.get('text') or ''}"
            for t in turns
        )[:9000]
        try:
            text, _ = self.client.complete(
                system=ATOMIC_SYSTEM,
                user=ATOMIC_PROMPT.replace("{session_date}", session_date or "unknown")
                .replace("{speakers}", speakers or "two speakers")
                .replace("{transcript}", transcript),
                max_tokens=2000,
            )
            import re

            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                raise ValueError("no JSON array in output")
            facts = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
            facts = facts[:40]
        except Exception as e:  # noqa: BLE001
            logger.warning("atomic extraction failed for %s: %s", session_id, str(e)[:120])
            facts = []
        self._cache[session_id] = facts
        self._flush()
        return facts

    @staticmethod
    def fact_id(user_id: str, session_id: str, fact: str) -> str:
        raw = f"{user_id}|{session_id}|{fact}"
        return "at-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
