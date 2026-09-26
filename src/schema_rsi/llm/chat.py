"""OpenAI-compatible Chat 薄客户端（评测 Answerer 使用；Mem0 内部 LLM 由其自己配置）。"""

from __future__ import annotations

from openai import OpenAI

from schema_rsi.config import Settings, get_settings


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> tuple[str, dict]:
        """返回 (回答文本, usage 信息)。

        注意：推理型模型（如 glm-5.3-flash）的思考 token 也计入 completion，
        max_tokens 过小会导致 content 为空，这里设下限 256。
        """
        max_tokens = max(int(max_tokens), 256)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},  # 关闭深度思考（见 mem0_llm.py）
        )
        content = resp.choices[0].message.content or "" if resp.choices else ""
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return content.strip(), usage


def make_chat_client(settings: Settings | None = None) -> ChatClient:
    s = settings or get_settings()
    if not (s.llm.base_url and s.llm.api_key and s.llm.model):
        raise RuntimeError(
            "LLM 未配置：请在 .env 中设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL（参考 .env.example）"
        )
    return ChatClient(
        base_url=s.llm.base_url,
        api_key=s.llm.api_key,
        model=s.llm.model,
        temperature=s.llm.temperature,
    )
