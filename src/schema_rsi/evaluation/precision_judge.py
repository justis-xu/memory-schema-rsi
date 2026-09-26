"""LLM 严格精准判分（按题型，默认 glm-5.3 非 flash 档）。

对宽松 judge 判 CORRECT 的题做第二道分级：
- exact:   全对且精准——gold 的每个元素都在、精度不低于 gold、无含糊措辞
- partial: 基本对但不精准——缺一个元素、粒度不足（对月错日）、正确答案被
  含糊表述（"I think X"）、在正确答案旁列举了错误备选
- wrong:   不含正确答案

设计要点：
- 判分输入只有 题目/gold/pred，与产生答案的组别无关（盲判）
- 结果按 (model, category, question, gold, pred) 哈希落盘缓存，断点续跑零成本
- 失败重试有限次；判不出抛 PrecisionJudgeError 由上层记录
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.chat import ChatClient

logger = logging.getLogger(__name__)

DEFAULT_PRECISION_MODEL = "glm-5.3"

VALID_GRADES = {"exact", "partial", "wrong"}


class PrecisionJudgeError(RuntimeError):
    """判分调用或解析失败。"""


_CATEGORY_RULES = {
    "temporal": """Category: TEMPORAL (date/time reasoning).
- Gold is a date, duration, or date-anchored event. "exact" requires the same point/period at the SAME granularity as gold: if gold gives a specific day, the day must be exactly right; a month-only gold requires the right month; "summer 2022" is matched by any date within that season; a duration gold ("6 months") requires the exact value.
- "Around/roughly/approximately <the right date>" is NOT exact (hedged precision) — grade partial.
- Right month but wrong day, or right year but wrong month: partial (close) or wrong (far) by your judgment.
- For counting questions ("how many times"), exact requires the exact final count; a count of a subset (e.g., "three completed" when asked how many written, total four) is wrong if the final count differs.""",
    "multi-hop": """Category: MULTI-HOP (answer synthesized across multiple conversations).
- Gold often has multiple elements (a list, or a composite fact). "exact" requires ALL elements present and correct. Missing exactly one element of a multi-element gold (≥2 elements) → partial. Missing more than one, or any element factually wrong → wrong.
- Naming the right entity type but not the specific entity ("a book Caroline recommended" when gold is the title "Becoming Nicole") is partial at best — the specific value is the answer.""",
    "single-hop": """Category: SINGLE-HOP (direct fact lookup).
- "exact" requires the precise fact: the exact name/number/place as in gold. A broader category containing the gold ("a car" for "Toyota Camry", "a city in Italy" for "Florence") is partial. A different specific value is wrong.
- For list golds ("X, Y, Z"), ALL items are required for exact; missing one → partial.""",
    "open-domain": """Category: OPEN-DOMAIN (opinion/inference).
- Gold states a conclusion, often qualified ("Likely no; ..."). "exact" requires the same conclusion at the same level of commitment, with the gold's key supporting reasoning if gold includes it. Right conclusion but weaker/stronger commitment or different stated reason → partial.""",
    "adversarial": """Category: ADVERSARIAL (gold: the conversation does NOT contain the answer).
- "exact" = the response clearly and cleanly indicates the information is not available/remembered, without inventing any specific answer.
- "partial" = indicates unavailability BUT also offers a specific guess (a name/date/fact not grounded).
- "wrong" = provides a specific fabricated answer as the answer.""",
}

_PROMPT_TEMPLATE = """You are grading answer PRECISION — an answer must be not just topically correct but exact and complete. Grade strictly: vague or partially-complete answers must NOT get "exact".

## Question
{question}

## Gold answer
{gold}

## Generated answer
{response}

## Category rules
{category_rules}

## General strictness rules
- "exact": every element of the gold answer is present and correct, at no less precision than the gold itself. Extra correct detail is fine. Enumerating alternatives ("either X or Y") is not exact. Hedged statements ("I think X", "probably X", "around X") are not exact when the gold is definitive.
- "partial": essentially on-topic and partially correct — misses one element of a multi-element gold, is less specific than gold, hedges the right answer, or mixes the right answer with wrong alternatives.
- "wrong": does not contain the correct answer.
- If the gold itself is approximate ("approximately summer 2022"), matching at that same approximation level IS exact.

Return JSON only: {{"grade": "exact"|"partial"|"wrong", "missing": ["missing or imprecise elements"], "reason": "one short sentence"}}"""


class PrecisionJudge:
    def __init__(
        self,
        settings: Settings | None = None,
        model: str = DEFAULT_PRECISION_MODEL,
        cache_path: str = "data/precision_judge_cache.json",
    ):
        self.settings = settings or get_settings()
        self.model = model
        self.client = ChatClient(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key,
            model=model,
        )
        self.cache_path = self.settings.resolve_path(cache_path)
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load_cache()

    # -- 缓存 ---------------------------------------------------------------
    def _load_cache(self) -> None:
        try:
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._cache = {}

    def flush(self) -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
            )

    @staticmethod
    def _key(model: str, category: str, question: str, gold: str, response: str) -> str:
        raw = json.dumps([model, category, question, gold, response], ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    # -- 判分 ---------------------------------------------------------------
    def grade(
        self, category: str | None, question: str, gold: str, response: str
    ) -> dict:
        """返回 {grade, missing, reason, cached}。adversarial 的空 gold 显示为无答案。"""
        cat = category or "open-domain"
        gold_display = gold or "(no answer — the conversation does not contain this information)"
        key = self._key(self.model, cat, question, gold_display, response or "")
        with self._lock:
            if key in self._cache:
                hit = dict(self._cache[key])
                hit["cached"] = True
                return hit

        rules = _CATEGORY_RULES.get(cat, _CATEGORY_RULES["open-domain"])
        user = _PROMPT_TEMPLATE.format(
            question=question,
            gold=gold_display,
            response=response or "(empty)",
            category_rules=rules,
        )
        last_err = ""
        for attempt in range(3):
            try:
                text, _ = self.client.complete(
                    system="You are a strict precision grader for conversational memory QA. Return JSON only.",
                    user=user,
                    max_tokens=300,
                )
                parsed = self._parse(text)
                with self._lock:
                    self._cache[key] = parsed
                out = dict(parsed)
                out["cached"] = False
                return out
            except Exception as e:  # noqa: BLE001 — 统一转成判分错误
                last_err = str(e)[:160]
                logger.warning("precision judge attempt %d failed: %s", attempt + 1, last_err)
        raise PrecisionJudgeError(f"precision judge failed after retries: {last_err}")

    @staticmethod
    def _parse(text: str) -> dict:
        import re

        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                grade = str(obj.get("grade", "")).lower()
                if grade in VALID_GRADES:
                    return {
                        "grade": grade,
                        "missing": [str(x) for x in (obj.get("missing") or [])][:6],
                        "reason": str(obj.get("reason", ""))[:200],
                    }
            except Exception:
                pass
        low = text.lower()
        for g in ("exact", "partial", "wrong"):
            if re.search(rf"\b{g}\b", low):
                return {"grade": g, "missing": [], "reason": text.strip()[:200]}
        raise PrecisionJudgeError(f"unparseable precision verdict: {text[:160]}")
