"""Rerank 薄客户端（阿里百炼）。

端点格式在运行时自适应：
1. 先尝试 OpenAI/Jina/Cohere 风格的 `POST {base_url}/rerank`（body: model/query/documents/top_n）；
2. 失败则回退到 DashScope 原生 text-rerank API。
两者都失败时由调用方决定是否降级为原始检索顺序（见 retriever.py）。
"""

from __future__ import annotations

import requests

_DASHSCOPE_NATIVE_RERANK = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


class RerankError(RuntimeError):
    pass


class RerankClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._last_endpoint: str | None = None

    @property
    def last_endpoint(self) -> str | None:
        """调试信息：最近一次成功使用的端点风格。"""
        return self._last_endpoint

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        """返回 [{"index": i, "score": s}, ...]，按相关性降序；index 指向 documents 下标。"""
        if not documents:
            return []
        top_n = top_n or len(documents)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 1) OpenAI 风格 /rerank
        try:
            resp = requests.post(
                f"{self.base_url}/rerank",
                headers=headers,
                json={"model": self.model, "query": query, "documents": documents, "top_n": top_n},
                timeout=self.timeout,
            )
            if resp.ok:
                results = resp.json().get("results", [])
                self._last_endpoint = "openai-style /rerank"
                return [
                    {
                        "index": int(x["index"]),
                        "score": float(x.get("relevance_score", x.get("score", 0.0)) or 0.0),
                    }
                    for x in results
                ]
            openai_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:  # pragma: no cover
            openai_err = str(e)

        # 2) DashScope 原生 text-rerank
        try:
            resp = requests.post(
                _DASHSCOPE_NATIVE_RERANK,
                headers=headers,
                json={
                    "model": self.model,
                    "input": {"query": query, "documents": documents},
                    "parameters": {"return_documents": False, "top_n": top_n},
                },
                timeout=self.timeout,
            )
            if resp.ok:
                out = resp.json().get("output", {}).get("results", [])
                self._last_endpoint = "dashscope native text-rerank"
                return [
                    {"index": int(x["index"]), "score": float(x.get("relevance_score", 0.0) or 0.0)}
                    for x in out
                ]
            raise RerankError(
                f"rerank failed on both endpoints; /rerank -> {openai_err}; "
                f"native -> HTTP {resp.status_code}: {resp.text[:200]}"
            )
        except requests.RequestException as e:
            raise RerankError(f"rerank request error: {e}; /rerank -> {openai_err}") from e
