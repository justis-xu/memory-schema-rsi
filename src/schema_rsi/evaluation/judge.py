"""LLM-judge 判分（复核口径）。

词面指标（token-F1 / contains）对日期词序、同义改写偏保守（例如
"19 January, 2023" vs "January 19, 2023" 判 miss 但语义完全正确），
judge 用 LLM 做语义等价判定作为第二口径。

注意：judge 默认复用 LLM_*（glm-5.3-flash，思考已关）；.env 的 JUDGE_*
配置了更强模型时优先使用（建议 judge 用比 answerer 强的档位）。
"""

from __future__ import annotations

import logging

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.chat import ChatClient

logger = logging.getLogger(__name__)

class JudgeError(RuntimeError):
    """判分服务或返回格式失败；该题不能计入准确率。"""

JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair QA grader. Compare the predicted answer against the gold "
    "answer for the question. Semantically equivalent answers are CORRECT, including "
    "different date formats/word order (e.g. '19 January, 2023' == 'January 19, 2023'), "
    "or a prediction that contains the gold answer with extra correct detail. "
    "Missing, wrong, or contradictory information is WRONG. "
    "Reply with EXACTLY one word on the first line: CORRECT or WRONG."
)


def make_judge_client(settings: Settings | None = None) -> ChatClient | None:
    """优先 JUDGE_*（未配置则复用 LLM_*）；两者都没有则返回 None。"""
    s = settings or get_settings()
    base_url = s.judge.base_url or s.llm.base_url
    api_key = s.judge.api_key or s.llm.api_key
    model = s.judge.model or s.llm.model
    if not (base_url and api_key and model):
        return None
    return ChatClient(base_url=base_url, api_key=api_key, model=model)


def judge_answer(
    question: str,
    expected: str,
    predicted: str,
    client: ChatClient,
    category: str | None = None,
    preset: str = "official",
    benchmark: str | None = None,
) -> tuple[bool, str]:
    """返回 (是否正确, judge 备注)。异常抛出 JudgeError，供评测续跑。

    preset:
    - official: mem0 官方 benchmark 的宽松 judge（部分正确即 CORRECT、日期 ±14 天
      容差等，JSON 输出；open-domain 的 gold 取分号前半）——与官方分数可比的口径
    - custom:   自研严格 CORRECT/WRORD
    """
    try:
        if preset == "official" and benchmark == "longmemeval_s":
            # LongMemEval 官方 judge：yes/no（<judge_thinking> 后结论）
            from schema_rsi.evaluation.prompts_lme_official import JUDGE_PROMPT_LME

            user = JUDGE_PROMPT_LME.format(question=question, answer=expected, response=predicted or "(empty)")
            text, _ = client.complete(
                system="You are evaluating conversational AI memory recall.", user=user, max_tokens=512
            )
            verdict_part = text.split("</judge_thinking>")[-1] if "</judge_thinking>" in text else text
            v = verdict_part.strip().lower()
            if v.startswith("yes") or "\nyes" in v:
                return True, text.strip()[:120]
            if v.startswith("no") or "\nno" in v:
                return False, text.strip()[:120]
            raise JudgeError(f"unrecognized LongMemEval verdict: {text[:120]}")

        if preset == "official":
            import json as _json
            import re as _re

            from schema_rsi.evaluation.prompts_official import (
                JUDGE_PROMPT_OFFICIAL,
                JUDGE_SYSTEM_PROMPT_OFFICIAL,
                preprocess_answer_official,
            )

            gold = preprocess_answer_official(category, expected)
            user = JUDGE_PROMPT_OFFICIAL.format(
                question=question, answer=gold, response=predicted or "(empty)"
            )
            text, _ = client.complete(
                system=JUDGE_SYSTEM_PROMPT_OFFICIAL, user=user, max_tokens=256
            )
            label = ""
            m = _re.search(r"\{.*\}", text, _re.S)
            if m:
                try:
                    label = str(_json.loads(m.group(0)).get("label", "")).upper()
                except Exception:
                    pass
            if not label:
                upper = text.upper()
                if "WRONG" in upper and "CORRECT" not in upper.replace("WRONG", ""):
                    label = "WRONG"
                elif "CORRECT" in upper:
                    label = "CORRECT"
            if label not in {"CORRECT", "WRONG"}:
                raise JudgeError(f"unrecognized LoCoMo verdict: {text[:120]}")
            return label == "CORRECT", text.strip()[:120]

        user = (
            f"Question: {question}\n"
            f"Gold answer: {expected}\n"
            f"Predicted answer: {predicted or '(empty)'}\n"
            f"Verdict:"
        )
        text, _ = client.complete(system=JUDGE_SYSTEM_PROMPT, user=user, max_tokens=256)
        verdict = text.strip().splitlines()[0].strip().upper() if text.strip() else ""
        if verdict not in {"CORRECT", "WRONG"}:
            raise JudgeError(f"unrecognized verdict: {text[:120]}")
        return verdict == "CORRECT", text.strip()[:80]
    except Exception as e:
        logger.warning("judge failed: %s", str(e)[:120])
        if isinstance(e, JudgeError):
            raise
        raise JudgeError(str(e)) from e
