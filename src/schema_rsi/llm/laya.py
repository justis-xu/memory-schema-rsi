"""Laya 决策服务客户端（自部署，公网直连）。

用途（检索层的语义决策，补 rerank 对"问法不同但事实相关"的盲区）：
1. 问题路由：判断题型（列表/多跳/时间/单事实/观点）→ 动态调整图聚合参数
2. 候选过滤：对图召回的候选逐条判定"直接回答/支持背景/无关"，
   probabilities 作为软特征过滤+排序（与 rerank 分数互补）

红线（遵守服务文档）：choice 选项 ≤20 且带描述；布尔用双选项 choice；
单条 state ≤1000 字；并发 ≤4；多问合并同一请求。
缓存：路由结果按问题哈希落盘（跨轮零成本）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import requests

from schema_rsi.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_LAYA_BASE = "https://u1172328-qpd4-5f1abfb1.weste.seetacloud.com:8443"

_ROUTE_QUESTIONS = {
    "qtype": {
        "type": "choice",
        "instructions": "What kind of memory question is this?",
        "criteria": {
            "list_enumeration": "asks to list/enum ALL items (what X did Y buy/list every)",
            "multi_hop": "requires combining facts about different people/things",
            "temporal": "asks when/how long; answer is a date or duration",
            "single_fact": "one direct fact about one person/thing",
            "opinion_inference": "would/is likely questions needing reasoning",
            "other": "anything else",
        },
    },
    "needs_full_list": {
        "type": "choice",
        "instructions": "Does a complete answer require enumerating multiple items?",
        "criteria": {"yes": "the answer is a list or count of several items",
                     "no": "a single value answers it"},
    },
}

_RELEVANCE_QUESTIONS = {
    "relevance": {
        "type": "choice",
        "instructions": "Is this memory relevant to answering the question?",
        "criteria": {
            "directly_answers": "contains a fact that directly answers the question",
            "supporting": "context about the same person/topic that helps reasoning",
            "irrelevant": "about a different person, topic, or fact",
        },
    }
}

# Jev-Mem 四维候选评分（awesome-jev output-validation 模式：每维一个独立命题，
# 单请求并行评估零额外延迟；时效维度的时间差在 state 里用代码算好，不让模型比日期）
_4DIM_QUESTIONS = {
    "relevant": {"type": "noul",
                 "instructions": "This memory is about the same person/subject the question asks about."},
    "answers_need": {"type": "noul",
                     "instructions": "This memory states specific information the complete answer requires (a name, number, date, or list item)."},
    "current": {"type": "noul",
                "instructions": "This memory reflects the CURRENT state of its subject (not superseded by the newer event noted in the state)."},
    "adds_info": {"type": "noul",
                  "instructions": "This memory adds information not already in typical top results (it is not generic filler)."},
}


class LayaError(RuntimeError):
    pass


class LayaClient:
    def __init__(self, settings: Settings | None = None, base_url: str | None = None):
        self.settings = settings or get_settings()
        cfg = (self.settings.raw.get("laya") or {})
        self.base = (base_url or cfg.get("base_url") or DEFAULT_LAYA_BASE).rstrip("/")
        self._sem = threading.Semaphore(4)  # 红线：并发 ≤4
        self._route_cache: dict[str, dict] = {}
        self._cache_path = Path(self.settings.resolve_path("data/search/laya_route_cache.json"))
        try:
            self._route_cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._route_cache = {}

    # ---- 基础调用 ----------------------------------------------------------

    def decide(self, state_body: str, questions: dict) -> dict:
        """POST /v1/decisions。返回 {"answers": {...}, "routing": {...}}。"""
        last = ""
        for attempt in range(3):
            try:
                with self._sem:
                    r = requests.post(
                        f"{self.base}/v1/decisions",
                        json={"state": {"body": state_body[:1000]}, "questions": questions},
                        timeout=15,
                    )
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:120]}"
                if r.status_code in (422, 500):  # 格式错/引擎异常：不重试同参数
                    break
                time.sleep(1.5 * (attempt + 1))  # 503 预热
            except requests.RequestException as e:
                last = str(e)[:120]
                time.sleep(1.5 * (attempt + 1))
        raise LayaError(f"laya decide failed: {last}")

    # ---- 场景 1：问题路由（缓存） ----------------------------------------

    def route_question(self, question: str) -> dict:
        """返回 {qtype, needs_full_list, probs}。按问题哈希缓存。"""
        key = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
        if key in self._route_cache:
            return self._route_cache[key]
        out = self.decide(question, _ROUTE_QUESTIONS)
        a = out.get("answers", {})
        route = {
            "qtype": (a.get("qtype") or {}).get("choice", "other"),
            "needs_full_list": (a.get("needs_full_list") or {}).get("choice", "no"),
            "probs": {k: (v or {}).get("probabilities", {}) for k, v in a.items()},
        }
        self._route_cache[key] = route
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._route_cache, ensure_ascii=False), encoding="utf-8")
        return route

    # ---- 场景 2：候选相关性（软特征） ------------------------------------

    def score_candidate(self, question: str, candidate: str) -> dict:
        """返回 {label, probs}，label ∈ directly_answers/supporting/irrelevant。"""
        state = f"Question: {question}\n\nMemory: {candidate}"
        out = self.decide(state, _RELEVANCE_QUESTIONS)
        a = (out.get("answers") or {}).get("relevance", {})
        return {
            "label": a.get("choice", "irrelevant"),
            "probs": a.get("probabilities", {}),
        }

    # ---- 场景 3：Jev 四维并行评分（一次请求四个命题） -----------------------

    def score_candidate_4dim(self, question: str, candidate: str) -> dict:
        """四维 noul 概率：relevant/answers_need/current/adds_info（0-1）。

        日期上下文由调用方算好拼进 state（Jev 弱项：不让模型自己比日期）。
        """
        state = f"Question: {question}\n\nMemory: {candidate}"
        out = self.decide(state, _4DIM_QUESTIONS)
        answers = out.get("answers") or {}
        probs = {}
        for k, v in answers.items():
            p = (v or {}).get("noul")
            probs[k] = float(p) if isinstance(p, (int, float)) else 0.5
        return probs

    def evidence_sufficient(self, question: str, context_digest: str) -> float:
        """证据充足判定（Jev-Mem 的停止准则）：返回 [0,1]。"""
        qs = {"sufficient": {"type": "noul",
                             "instructions": "The context above contains ALL specific facts needed to answer the question completely and precisely (every list item, exact date, and exact value)."}}
        state = self.evidence_state(question, context_digest)
        out = self.decide(state, qs)
        p = ((out.get("answers") or {}).get("sufficient") or {}).get("noul")
        return float(p) if isinstance(p, (int, float)) else 0.5

    @staticmethod
    def evidence_state(question: str, context_digest: str) -> str:
        """Return the exact state body sent to /v1/decisions for evidence scoring."""
        return f"Question: {question}\n\nContext:\n{context_digest[:950]}"[:1000]
