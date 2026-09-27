"""统一评测管线骨架。

    BenchmarkCase
         ↓ ingest（history → Mem0，per_case user_id 隔离）
         ↓ retrieve（Mem0 向量检索，可选 rerank）
         ↓ GraphRetriever（graph_enabled=true 时）
         ↓ Answerer（LLM）
         ↓ Evaluator（token-F1 / contains）
         ↓ EvaluationResult → JSONL

graph_enabled 是对照实验开关：Mem0 Only（false）vs Mem0 + Graph（true）。
"""

from __future__ import annotations

import logging
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from schema_rsi.benchmarks.base import BenchmarkCase
from schema_rsi.config import Settings, get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.evaluator import evaluate_answers
from schema_rsi.evaluation.result import EvaluationResult, append_jsonl, summarize
from schema_rsi.evaluation.retriever import GraphRetriever, Mem0Retriever
from schema_rsi.memory.base import MemoryBackend, MemoryRecord

logger = logging.getLogger(__name__)


def _jev_context_snapshot(memories: list[MemoryRecord]) -> list[dict]:
    """Freeze the exact ordered records seen by the answerer at a Jev decision."""
    return [
        {
            "id": m.id,
            "content": m.content,
            "content_sha256": hashlib.sha256((m.content or "").encode()).hexdigest(),
            "session_date": (m.metadata or {}).get("session_date"),
            "session_id": (m.metadata or {}).get("session_id"),
            "via": (m.metadata or {}).get("via"),
        }
        for m in memories
    ]


def _jev_graph_pool_snapshot(memories: list[dict]) -> list[dict]:
    return [
        {"id": m.get("id"), "via": m.get("via"), "support": m.get("support"),
         "content_sha256": hashlib.sha256((m.get("content") or "").encode()).hexdigest()}
        for m in memories
    ]


class EvaluationPipeline:
    def __init__(
        self,
        backend: MemoryBackend,
        settings: Settings | None = None,
        graph_store=None,
        answerer: Answerer | None = None,
        judge_client=None,
    ):
        self.settings = settings or get_settings()
        self.backend = backend
        self.graph_store = graph_store
        self.answerer = answerer
        self.judge_client = judge_client  # 可选：用于 answer 多采样的多数票
        self.retriever = Mem0Retriever(backend, self.settings)
        self.graph_retriever = GraphRetriever(graph_store, self.settings) if graph_store else None

    # ---- ingest ----

    def ingest_case(self, case: BenchmarkCase, *, max_turns: int | None = None, user_id: str | None = None) -> int:
        """把 case 的 history（可截断）按 session 写入 Mem0，返回新增记忆条数。

        user_id 默认按 user_id_strategy 生成；同 conversation 的多个 QA 可显式传同一个
        user_id 共享一份记忆（避免重复 ingest 的 LLM 开销）。

        容错：按 session 粒度捕获失败（如 LLM 供应商内容过滤），跳过并记录到
        self.ingest_errors，不中断整体 ingest。
        """
        user_id = user_id or self.case_user_id(case)
        added = 0
        self.ingest_errors: list[dict] = getattr(self, "ingest_errors", [])
        remaining = max_turns
        for session in case.history:
            if remaining is not None and remaining <= 0:
                break
            turns = session.get("turns", [])
            if remaining is not None:
                turns = turns[:remaining]
                remaining -= len(turns)
            if not turns:
                continue
            messages = [{"role": t.get("role", "user"), "content": t.get("content", "")} for t in turns]
            # 注入会话日期锚（mem0 OSS 不支持 timestamp 参数，相对日期会错锚到"今天"）
            from schema_rsi.benchmarks.base import parse_session_date

            iso_date = parse_session_date(session.get("date"))
            if iso_date:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            f"Conversation date: {iso_date}. "
                            "Use this date to resolve relative time expressions like "
                            "'yesterday', 'next month', 'in three weeks'."
                        ),
                    }
                ] + messages
            try:
                records = self.backend.add_memory(
                    user_id,
                    messages,
                    metadata={
                        "benchmark": case.benchmark,
                        "case_id": case.case_id,
                        "session_id": str(session.get("session_id")),
                        "session_date": session.get("date"),
                        "user_id": user_id,
                    },
                )
                added += len(records)
            except Exception as e:  # 内容过滤/限流等：跳过该 session，继续
                self.ingest_errors.append(
                    {
                        "case_id": case.case_id,
                        "session_id": str(session.get("session_id")),
                        "error": str(e)[:200],
                    }
                )
                logger.warning(
                    "ingest 跳过 %s session=%s: %s", case.case_id, session.get("session_id"), str(e)[:120]
                )
        return added

    def case_user_id(self, case: BenchmarkCase) -> str:
        return self.settings.make_user_id(case.benchmark, case.case_id)

    # ---- evaluate ----

    def evaluate_case(
        self,
        case: BenchmarkCase,
        *,
        graph_enabled: bool | None = None,
        user_id: str | None = None,
        answer_samples: int = 1,
    ) -> EvaluationResult:
        use_graph = self.settings.graph_enabled if graph_enabled is None else graph_enabled
        if use_graph and self.graph_retriever is None:
            raise ValueError("graph_enabled=true 但 pipeline 未注入 graph_store")

        user_id = user_id or self.case_user_id(case)
        # 官方 answer prompt 需要 reference_date（对话最后会话日期）做时间锚
        from schema_rsi.benchmarks.base import parse_session_date

        reference_date = None
        if case.history:
            reference_date = parse_session_date(case.history[-1].get("date"))
        latency: dict = {}

        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(case.question, user_id=user_id)
        if any((r.metadata or {}).get("user_id") != user_id for r in retrieved):
            raise ValueError("memory retrieval returned a different or unknown user_id")
        latency["retrieve"] = round(time.perf_counter() - t0, 3)
        latency["rerank_fallback"] = bool(self.retriever.last_rerank_error)

        context_budget = min(self._max_context(), len(retrieved))
        graph_mode = str(self.evaluation_cfg().get("graph_mode", "bypass"))
        graph_slots = min(max(0, int(self.evaluation_cfg().get("graph_slots", 3))), context_budget)
        graph_seed_k = max(0, int(self.evaluation_cfg().get("graph_seed_k", 5)))
        graph_memories: list[dict] = []
        fused_pool_size = None
        laya_route_info: dict | None = None
        laya_filter_info: dict | None = None
        if use_graph and graph_seed_k:
            t0 = time.perf_counter()
            if graph_mode == "fused":
                per_node = int(self.evaluation_cfg().get("fused_per_node", 15))
                pool = int(self.evaluation_cfg().get("fused_pool", 30))
                # Laya 路由：按题型动态调聚合参数（列表/多跳要宽，单事实要省）
                if getattr(self, "laya_route", False):
                    try:
                        route = self._laya().route_question(case.question)
                        per_node, pool = self._route_params(route, per_node, pool)
                        laya_route_info = {"qtype": route["qtype"],
                                           "needs_full_list": route["needs_full_list"],
                                           "per_node": per_node, "pool": pool}
                    except Exception as e:  # noqa: BLE001
                        logger.warning("laya route failed, keep defaults: %s", str(e)[:100])
                candidates = self.graph_retriever.retrieve_fused(  # type: ignore[union-attr]
                    retrieved[:graph_seed_k],
                    exclude_ids={r.id for r in retrieved},
                    per_node_limit=per_node,
                    max_candidates=pool,
                    hops=getattr(self, "fused_hops", None),
                )
                graph_memories = [g for g in candidates if g.get("user_id") == user_id]
                # Laya 过滤：对图候选做问题条件的相关性判定，滤无关、按软概率+support 排序
                if getattr(self, "laya_filter", False) and graph_memories:
                    try:
                        graph_memories, laya_filter_info = self._laya_filter(
                            case.question, graph_memories
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("laya filter failed, keep all: %s", str(e)[:100])
                fused_pool_size = len(retrieved) + len(graph_memories)
            elif graph_slots:
                candidates = self.graph_retriever.retrieve(  # type: ignore[union-attr]
                    retrieved[:graph_seed_k], max_total=max(15, graph_slots * 5)
                )
                graph_memories = [g for g in candidates if g.get("user_id") == user_id]
            latency["graph"] = round(time.perf_counter() - t0, 3)

        if self.answerer is None:
            self.answerer = Answerer(self.settings)

        graph_context_ids: list[str] = []
        if use_graph and graph_mode == "fused" and graph_memories:
            # 统一重排：向量结果 + 图聚合候选同池竞争，预算与基线臂一致；
            # rerank 失败则降级回纯向量（等价基线口径，记录于 latency）
            # 注意：不能叫 pool——会遮蔽上面的整型 fused_pool，
            # 导致 _evidence_expand 里 pool * 1.5 变 list×float 而崩溃
            pool_records = list(retrieved)
            for g in graph_memories:
                if g.get("id") and g.get("content"):
                    pool_records.append(
                        MemoryRecord(id=g["id"], content=g["content"],
                                     metadata={"via": g.get("via"), "user_id": g["user_id"]})
                    )
            reranked = self.retriever.rerank_pool(case.question, pool_records, top_n=context_budget)
            if reranked is not None:
                answer_memories = reranked
                graph_ids = {g["id"] for g in graph_memories}
                graph_context_ids = [m.id for m in answer_memories if m.id in graph_ids]
            else:
                answer_memories = list(retrieved[:context_budget])
        else:
            # bypass 模式：图候选替换尾部向量记忆；两臂的回答上下文上限相同。
            answer_memories = list(retrieved[:context_budget])
            if use_graph and graph_memories:
                seen_ids = {r.id for r in retrieved}
                additions: list[MemoryRecord] = []
                for g in graph_memories:
                    if (g.get("user_id") == user_id and g.get("id")
                            and g["id"] not in seen_ids and g.get("content")):
                        seen_ids.add(g["id"])
                        additions.append(
                            MemoryRecord(id=g["id"], content=g["content"],
                                         metadata={"via": g.get("via"), "user_id": g["user_id"]})
                        )
                        graph_context_ids.append(g["id"])
                        if len(additions) >= graph_slots:
                            break
                if additions:
                    answer_memories = list(retrieved[:context_budget - len(additions)]) + additions

        # Jev-Mem 停止准则：证据不足 → 聚合宽度加倍重拉一次（仅 fused 模式）
        jev_stop_info: dict | None = None
        if (getattr(self, "jev_stop", False) and use_graph and graph_mode == "fused"
                and graph_seed_k):
            from schema_rsi.llm.laya import LayaClient

            first_context = _jev_context_snapshot(answer_memories)
            first_pool = _jev_graph_pool_snapshot(graph_memories)
            digest = " | ".join((m.content or "")[:60] for m in answer_memories[:15])
            jev_stop_info = {
                "trace_version": 1,
                "first_context": first_context,
                "first_graph_pool": first_pool,
                "first_decision_state": LayaClient.evidence_state(case.question, digest),
                "first_digest_sha256": hashlib.sha256(digest.encode()).hexdigest(),
            }
            try:
                laya = self._laya()
                p_suff = laya.evidence_sufficient(case.question, digest)
                jev_stop_info["first_raw"] = p_suff
                jev_stop_info["first"] = round(p_suff, 2)
                if p_suff < 0.4:
                    answer_memories, graph_memories = self._evidence_expand(
                        case, user_id, retrieved, graph_memories, per_node, pool, context_budget)
                    digest2 = " | ".join((m.content or "")[:60] for m in answer_memories[:15])
                    jev_stop_info.update({
                        "expanded": True,
                        "second_context": _jev_context_snapshot(answer_memories),
                        "second_graph_pool": _jev_graph_pool_snapshot(graph_memories),
                        "second_decision_state": LayaClient.evidence_state(case.question, digest2),
                        "second_digest_sha256": hashlib.sha256(digest2.encode()).hexdigest(),
                    })
                    p2 = laya.evidence_sufficient(case.question, digest2)
                    jev_stop_info["second_raw"] = p2
                    jev_stop_info["second"] = round(p2, 2)
                else:
                    jev_stop_info["expanded"] = False
            except Exception as e:  # noqa: BLE001
                jev_stop_info["error_type"] = type(e).__name__
                logger.warning("jev_stop failed: %s", str(e)[:100])

        if use_graph and graph_mode == "fused":
            final_graph_ids = {g.get("id") for g in graph_memories}
            graph_context_ids = [m.id for m in answer_memories if m.id in final_graph_ids]

        # Answer：可选多采样（self-consistency）——LLM 作答方差 ±1~2 题，
        # 采样多次后由 judge 多数票选出最终答案，显著稳定结果
        n_samples = max(1, int(answer_samples))
        t0 = time.perf_counter()
        answers: list[str] = []
        for i in range(n_samples):
            temp = 0.7 if n_samples > 1 else None  # 单次保持 temperature=0
            text, _ = self.answerer.answer(
                case.question,
                answer_memories,
                temperature=temp,
                reference_date=reference_date,
                question_date=(case.metadata or {}).get("question_date"),
                benchmark=case.benchmark,
            )
            answers.append(text)
        latency["answer"] = round(time.perf_counter() - t0, 3)

        predicted = answers[0]
        votes: list[bool] = []
        if self.judge_client is not None:
            from schema_rsi.evaluation.judge import judge_answer

            judge_preset = (self.settings.evaluation or {}).get("judge_prompt_preset", "official")
            if n_samples == 1:
                # 单采样：直接对最终答案判一次
                ok, _ = judge_answer(
                    case.question, case.answer, predicted, self.judge_client,
                    category=case.category, preset=judge_preset, benchmark=case.benchmark,
                )
                votes.append(ok)
            else:
                for ans in answers:
                    ok, _ = judge_answer(
                        case.question, case.answer, ans, self.judge_client,
                        category=case.category, preset=judge_preset, benchmark=case.benchmark,
                    )
                    votes.append(ok)
                if any(votes):  # 多数票/优先正确样本；平票取第一个
                    predicted = next(a for a, v in zip(answers, votes) if v)
        metrics = evaluate_answers(case.answer, predicted)
        if votes:
            metrics["judge_votes"] = votes
            metrics["judge_correct"] = sum(votes) * 2 > len(votes)
        return EvaluationResult(
            case_id=case.case_id,
            benchmark=case.benchmark,
            question=case.question,
            expected_answer=case.answer,
            predicted_answer=predicted,
            retrieved_memories=retrieved,
            graph_memories=graph_memories,
            metrics=metrics,
            latency=latency,
            metadata={
                "category": case.category,
                "user_id": user_id,
                "graph_enabled": use_graph,
                "answer_context_ids": [m.id for m in answer_memories],
                "answer_context_budget": context_budget,
                "graph_context_ids": graph_context_ids,
                "graph_seed_k": graph_seed_k if use_graph else None,
                "graph_slots": graph_slots if use_graph else None,
                "graph_mode": graph_mode if use_graph else None,
                "fused_pool_size": fused_pool_size,
                "fused_graph_in_context": len(graph_context_ids) if graph_mode == "fused" and use_graph else None,
                "laya_route": laya_route_info,
                "laya_filter": laya_filter_info,
                "jev_stop": jev_stop_info,
                "atomic_pool": bool(getattr(self.retriever, "use_atomic", False))
                and bool(getattr(self.retriever, "atomic_col", None)),
                "rerank_used": self.retriever._reranker is not None and not self.retriever.last_rerank_error,
                "answerer_model": self.settings.llm.model,
                "answer_samples": n_samples,
                "answer_candidates": answers if n_samples > 1 else None,
            },
        )

    def _max_context(self) -> int:
        return int(self.evaluation_cfg().get("max_context_memories", 10))

    # ---- Laya 决策服务接入 ----

    def _laya(self):
        if getattr(self, "_laya_client", None) is None:
            from schema_rsi.llm.laya import LayaClient
            self._laya_client = LayaClient(self.settings)
        return self._laya_client

    @staticmethod
    def _route_params(route: dict, per_node: int, pool: int) -> tuple[int, int]:
        """题型 → (聚合宽度, 候选池)。列表/多跳要宽（宁可多带），单事实要省。"""
        qtype = route.get("qtype", "other")
        if route.get("needs_full_list") == "yes" or qtype in ("list_enumeration", "multi_hop"):
            return max(per_node, 30), max(pool, 45)
        if qtype == "temporal":
            return max(per_node, 25), max(pool, 35)
        if qtype == "single_fact":
            return min(per_node, 12), min(pool, 24)
        return per_node, pool

    def _laya_filter(self, question: str, candidates: list[dict]) -> tuple[list[dict], dict]:
        """图候选 Jev 四维过滤：relevant/answers_need 双低者丢弃；按软分+support 排序。"""
        laya = self._laya()
        scored = []
        for g in candidates[:10]:  # 上限 10 个（Laya 多问合并，单候选一次请求）
            p = laya.score_candidate_4dim(question, g.get("content") or "")
            if p.get("relevant", 0.5) < 0.35 and p.get("answers_need", 0.5) < 0.35:
                continue
            soft = min(p.get("relevant", 0.5), p.get("answers_need", 0.5))
            scored.append((soft, p, g))
        max_support = max((g.get("support", 1) for _, _, g in scored), default=1) or 1
        scored.sort(key=lambda t: -(t[0] + 0.1 * (t[2].get("support", 0) / max_support)))
        kept = [{**g, "laya": round(soft, 3)} for soft, p, g in scored]
        info = {"kept": len(kept), "dropped": len(candidates[:10]) - len(kept)}
        return kept, info

    def _evidence_expand(self, case, user_id: str, retrieved, graph_memories: list[dict],
                         per_node: int, pool: int, context_budget: int):
        """证据不足时的一次性扩拉：聚合宽度加倍，重建池并重排（Jev-Mem 停止准则的逆用）。"""
        candidates = self.graph_retriever.retrieve_fused(  # type: ignore[union-attr]
            retrieved[:max(5, int(self.evaluation_cfg().get("graph_seed_k", 5)))],
            exclude_ids={r.id for r in retrieved},
            per_node_limit=per_node * 2,
            max_candidates=int(pool * 1.5),
            hops=getattr(self, "fused_hops", None),
        )
        graph_memories = [g for g in candidates if g.get("user_id") == user_id]
        pool_records = list(retrieved) + [
            MemoryRecord(id=g["id"], content=g["content"],
                         metadata={"via": g.get("via"), "user_id": g["user_id"]})
            for g in graph_memories if g.get("id") and g.get("content")
        ]
        reranked = self.retriever.rerank_pool(case.question, pool_records, top_n=context_budget)
        if reranked is None:
            reranked = list(retrieved[:context_budget])
        return reranked, graph_memories

    def evaluation_cfg(self) -> dict:
        return self.settings.evaluation or {}

    # ---- run ----

    def run(
        self,
        cases: list[BenchmarkCase],
        *,
        graph_enabled: bool | None = None,
        ingest: bool = True,
        max_turns: int | None = None,
        write_jsonl: bool = True,
        user_id: str | None = None,
    ) -> list[EvaluationResult]:
        def _one(case: BenchmarkCase) -> EvaluationResult:
            if ingest:
                added = self.ingest_case(case, max_turns=max_turns, user_id=user_id)
                logger.info("ingested %s (+%d memories)", case.case_id, added)
            return self.evaluate_case(case, graph_enabled=graph_enabled, user_id=user_id)

        max_workers = int(self._cfg().get("max_workers", 1) or 1)
        if max_workers > 1 and len(cases) > 1:
            # case 级并行：各 case 的 user_id/ingest/QA 相互独立（per_case 策略下完全隔离）；
            # pool.map 保持结果与 cases 同序，JSONL 输出与串行版逐行可比
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(_one, cases))
        else:
            results = [_one(case) for case in cases]
        if write_jsonl and results:
            use_graph = results[0].metadata.get("graph_enabled", False)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = (
                self.settings.results_dir
                / f"eval_{stamp}_graph{'on' if use_graph else 'off'}.jsonl"
            )
            append_jsonl(results, path)
            logger.info("results written to %s", path)
        return results

    @staticmethod
    def print_summary(label: str, results: list[EvaluationResult]) -> None:
        s = summarize(results)
        print(f"[{label}] cases={s['cases']} avg_token_f1={s.get('avg_token_f1')} "
              f"contains_rate={s.get('contains_rate')} avg_answer_s={s.get('avg_answer_latency_s')}")
