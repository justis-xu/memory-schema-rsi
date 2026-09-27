"""Verifier verdicts cannot commit an answer without a resolvable support ID."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema_rsi.memory.base import MemoryRecord  # noqa: E402
from schema_rsi_lab.verify import repair_case, verify_answer  # noqa: E402


class StubClient:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, **_kwargs):
        return json.dumps(next(self.replies), ensure_ascii=False), {}


def test_supported_verdict_needs_resolvable_support_index():
    record = MemoryRecord(id="m1", content="梅拉妮最近和孩子们一起画画", metadata={})
    for indices in ([], [999], ["bad"]):
        verdict = verify_answer(StubClient([{"status": "supported", "support_indices": indices}]),
                                "梅拉妮喜欢做什么？", "画画", [record])
        assert verdict.status == "uncertain"
        assert verdict.support_ids == ()
    supported = verify_answer(StubClient([{"status": "supported", "support_indices": [1]}]),
                              "梅拉妮喜欢做什么？", "画画", [record])
    assert supported.status == "supported"
    assert supported.support_ids == ("m1",)


def test_repair_does_not_accept_unattributed_supported_reply():
    class Answerer:
        def answer(self, *_args, **_kwargs):
            return "画画", {}

    case = {"id": "x", "category": 1, "base_ranked": ["m1"],
            "baseline_answer": "陶艺", "baseline_correct": False,
            "question": "梅拉妮喜欢做什么？"}
    memories = {"m1": {"user_id": "u", "content": "梅拉妮最近和孩子们一起画画",
                        "metadata": {}}}
    verifier = StubClient([{"status": "contradicted", "support_indices": []},
                           {"status": "supported", "support_indices": [999]}])
    graph = SimpleNamespace(policy=SimpleNamespace(verify_enabled=True))
    row = repair_case(case, memories, graph, None, Answerer(), verifier,
                      reference_date=None)
    assert row["action"] == "same_context_reanswer"
    assert row["proposed_answer"] == "画画"
    assert row["second_verdict"]["status"] == "uncertain"
    assert row["accepted"] is False
    assert row["final_answer"] == "陶艺"

    valid_verifier = StubClient([{"status": "contradicted", "support_indices": []},
                                 {"status": "supported", "support_indices": [1]}])
    valid = repair_case(case, memories, graph, None, Answerer(), valid_verifier,
                        reference_date=None)
    assert valid["accepted"] is True
    assert valid["support_ids"] == ["m1"]
    assert valid["final_answer"] == "画画"
