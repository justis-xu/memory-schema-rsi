"""Answerer：基于检索到的记忆 + LLM 生成答案（OpenAI-compatible）。"""

from __future__ import annotations

import time

from schema_rsi.config import Settings, get_settings
from schema_rsi.llm.chat import ChatClient, make_chat_client
from schema_rsi.memory.base import MemoryRecord

ANSWER_SYSTEM_PROMPT = (
    "You are a precise QA assistant with long-term memory. "
    "Answer the question using ONLY the provided memories. "
    "Include ALL specific details that answer the question — exact dates (with the day "
    "when available), every named attribute, item or place the question asks about. "
    "Synthesize across multiple memories when needed (e.g. common attributes of two people). "
    "Reply with the answer itself only — short, no explanations, no preamble. "
    "If the memories do not contain the answer, reply with your best guess from the memories."
)


def build_user_prompt(question: str, memories: list[MemoryRecord]) -> str:
    lines = ["Memories:", "- (no memories retrieved)"] if not memories else ["Memories:"]
    if memories:
        lines = ["Memories:"]
        for i, m in enumerate(memories, 1):
            lines.append(f"{i}. {m.content}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


class Answerer:
    def __init__(self, settings: Settings | None = None, client: ChatClient | None = None):
        self.settings = settings or get_settings()
        self.client = client or make_chat_client(self.settings)
        # official = mem0 官方 benchmark 的 7 步 answer prompt；custom = 自研简版
        self.preset = (self.settings.evaluation or {}).get("answer_prompt_preset", "official")

    def answer(
        self,
        question: str,
        memories: list[MemoryRecord],
        temperature: float | None = None,
        reference_date: str | None = None,
        question_date: str | None = None,
        benchmark: str | None = None,
    ) -> tuple[str, dict]:
        t0 = time.perf_counter()
        if self.preset == "official" and benchmark == "longmemeval_s":
            # LongMemEval 官方：question_date 锚定 + newest-first 日期分组 + <mem_thinking>
            from schema_rsi.evaluation.prompts_lme_official import build_official_answer_prompt_lme

            prompt = build_official_answer_prompt_lme(question, memories, question_date)
            text, usage = self.client.complete(
                system="You are a personal assistant with access to memories from past conversations.",
                user=prompt, max_tokens=2000, temperature=temperature,
            )
            if "</mem_thinking>" in text:
                text = text.split("</mem_thinking>")[-1].strip() or text.strip()
        elif self.preset in ("official", "official_precise"):
            # LoCoMo 官方：7 步 answer prompt + reference_date
            # official_precise：官方模板 + 精准输出要求（100 轮探索的 answer 轴变体）
            from schema_rsi.evaluation.prompts_official import build_official_answer_prompt

            prompt = build_official_answer_prompt(question, memories, reference_date)
            if self.preset == "official_precise":
                prompt += (
                    "\n\n## PRECISION REQUIREMENTS (override any conflicting instruction above)\n"
                    "- Dates: state the maximum precision the memories support (the exact day when\n"
                    "  known). NEVER hedge with \"around/approximately/roughly\" unless the memory\n"
                    "  itself only supports that precision.\n"
                    "- Lists: include EVERY distinct item found across all memories; do not stop\n"
                    "  after the first few. Order does not matter.\n"
                    "- Counts: enumerate every instance, then give the exact final count.\n"
                    "- Prefer the specific name/title/place/number over any broader category.\n"
                    "- Answer with the fact itself first, in as few words as precision allows."
                )
            text, usage = self.client.complete(
                system="You are answering a question using retrieved memories from past conversations.",
                user=prompt, max_tokens=1500, temperature=temperature,
            )
            if "ANSWER:" in text:
                text = text.split("ANSWER:")[-1].strip() or text.strip()
        else:
            prompt = build_user_prompt(question, memories)
            text, usage = self.client.complete(
                system=ANSWER_SYSTEM_PROMPT, user=prompt, max_tokens=1024, temperature=temperature
            )
        info = {"answer_latency_s": round(time.perf_counter() - t0, 3), "usage": usage}
        return text, info
