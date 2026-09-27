"""Jev trace must preserve both decision contexts and final graph provenance."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_rsi.benchmarks.base import BenchmarkCase
from schema_rsi.config import get_settings
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from schema_rsi.llm.laya import LayaClient
from schema_rsi.memory.base import MemoryRecord


class FakeRetriever:
    last_rerank_error = None
    _reranker = object()

    def __init__(self):
        self.vector = [MemoryRecord("v1", "原有直接证据", {"user_id": "u", "session_date": "2023-01-01"}),
                       MemoryRecord("v2", "向量旁支", {"user_id": "u"})]

    def retrieve(self, question, user_id):
        return self.vector

    def rerank_pool(self, question, records, top_n):
        return [records[0], records[-1]]


class FakeGraphRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve_fused(self, *args, **kwargs):
        self.calls += 1
        mid = f"g{self.calls}"
        return [{"id": mid, "content": f"图证据{self.calls}", "user_id": "u",
                 "via": [f"entity:{mid}"], "support": 1}]


class FakeDecision:
    def __init__(self):
        self.digests = []

    def evidence_sufficient(self, question, digest):
        self.digests.append(digest)
        return [0.12345, 0.87654][len(self.digests) - 1]


class FakeAnswerer:
    def answer(self, *args, **kwargs):
        return "原有直接证据", {}


def test_jev_expansion_trace_and_final_graph_ids():
    settings = copy.deepcopy(get_settings())
    settings.evaluation = {"graph_enabled": True, "graph_mode": "fused",
                           "graph_seed_k": 1, "max_context_memories": 2,
                           "fused_per_node": 2, "fused_pool": 3}
    pipeline = EvaluationPipeline(backend=object(), settings=settings,
                                  graph_store=object(), answerer=FakeAnswerer())
    pipeline.retriever = FakeRetriever()
    pipeline.graph_retriever = FakeGraphRetriever()
    pipeline.jev_stop = True
    decision = FakeDecision()
    pipeline._laya = lambda: decision
    case = BenchmarkCase("case", "locomo", [], "问什么？", "原有直接证据")

    result = pipeline.evaluate_case(case, graph_enabled=True, user_id="u")
    trace = result.metadata["jev_stop"]

    assert trace["trace_version"] == 1
    assert [r["id"] for r in trace["first_context"]] == ["v1", "g1"]
    assert [r["id"] for r in trace["second_context"]] == ["v1", "g2"]
    assert trace["first_context"][0]["session_date"] == "2023-01-01"
    assert [r["id"] for r in trace["first_graph_pool"]] == ["g1"]
    assert [r["id"] for r in trace["second_graph_pool"]] == ["g2"]
    assert trace["first_decision_state"] == LayaClient.evidence_state(case.question, decision.digests[0])
    assert trace["second_decision_state"] == LayaClient.evidence_state(case.question, decision.digests[1])
    assert trace["first_raw"] == 0.12345 and trace["second_raw"] == 0.87654
    assert trace["first"] == 0.12 and trace["second"] == 0.88
    assert result.metadata["answer_context_ids"] == ["v1", "g2"]
    assert result.metadata["graph_context_ids"] == ["g2"]
    assert result.metadata["fused_graph_in_context"] == 1


def test_evidence_state_preserves_legacy_truncation():
    question = "问题" * 30
    digest = "证据" * 800
    legacy_state = f"Question: {question}\n\nContext:\n{digest[:950]}"[:1000]
    assert LayaClient.evidence_state(question, digest) == legacy_state
