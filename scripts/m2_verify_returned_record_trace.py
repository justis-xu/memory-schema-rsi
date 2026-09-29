#!/usr/bin/env python3
"""Verify optional outcome snapshots with real Chinese source and injected output facts."""

import copy
import hashlib
import json

from schema_rsi.benchmarks.base import BenchmarkCase
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from schema_rsi.memory.base import MemoryRecord

from m2_poc_lme_update_extraction import ROOT, PACKET

RUN = ROOT / "results/analysis/m2_lme_update_extraction_20260929.json"
OUT = ROOT / "results/analysis/m2_returned_record_trace_20260929.json"
QUESTION_ID = "031748ae"


class Recorder:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append(copy.deepcopy((user_id, messages, metadata)))
        return [copy.deepcopy(self.outputs[metadata["session_id"]])]


def replay(case, backend, sink=None):
    pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
    pipeline.backend = backend
    return pipeline.ingest_case(case, user_id="offline:" + case.case_id,
                                source_trace_sink=sink,
                                trace_run_id="source-output-replay",
                                trace_attempt_id="first")


def main():
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")
    packet_bytes = PACKET.read_bytes()
    packet_case = next(c for c in json.loads(packet_bytes)["cases"]
                       if c["question_id"] == QUESTION_ID)
    selected = {"answer_8748f791_1": [4, 10], "answer_8748f791_2": [2]}
    sessions = []
    for session in packet_case["source"]["chinese"]["answer_sessions"]:
        if session["session_id"] not in selected:
            continue
        turns = [t for t in session["turns"] if t["index"] in selected[session["session_id"]]]
        sessions.append({"session_id": session["session_id"], "date": session["date"],
                         "turns": [{"role": t["role"], "content": t["content"]} for t in turns]})
    assert len(sessions) == 2
    case = BenchmarkCase(QUESTION_ID, "longmemeval_s", sessions,
                         "gold question must not enter trace", "gold answer must not enter trace")
    run_bytes = RUN.read_bytes()
    run = json.loads(run_bytes)
    policy = next(c for c in run["calls"] if c["question_id"] == QUESTION_ID
                  and c["arm"] == "source_policy")
    facts = policy["parsed_response"]["facts"]
    fact_by_session = {"answer_8748f791_1": facts[1],
                       "answer_8748f791_2": facts[5]}
    outputs = {}
    for session in sessions:
        session_id = session["session_id"]
        fact = fact_by_session[session_id]
        content = f"{fact['subject']} {fact['relation']} {fact['object']}（{fact['time_scope']}）"
        outputs[session_id] = MemoryRecord(
            id="offline:" + session_id, content=content,
            metadata={"session_id": session_id, "session_date": session["date"],
                      "event": "ADD", "private_metadata": "must not be copied"})
    plain, traced, events = Recorder(outputs), Recorder(outputs), []
    assert replay(case, plain) == replay(case, traced, events.append) == 2
    assert plain.calls == traced.calls
    assert len(events) == 4
    pairs = []
    for input_event, outcome_event in zip(events[::2], events[1::2]):
        batch = input_event["batch"]
        snapshot = outcome_event["returned_records"][0]
        expected = outputs[batch["session_id"]]
        assert input_event["run_id"] == outcome_event["run_id"] == "source-output-replay"
        assert input_event["attempt_id"] == outcome_event["attempt_id"] == "first"
        assert batch["batch_id"] == outcome_event["batch_id"]
        assert snapshot["id"] == expected.id and snapshot["content"] == expected.content
        assert snapshot["content_sha256"] == hashlib.sha256(expected.content.encode()).hexdigest()
        assert snapshot["session_id"] == batch["session_id"]
        assert "private_metadata" not in snapshot
        pairs.append({"batch_id": batch["batch_id"],
                      "session_id": batch["session_id"],
                      "source_turn_count": len(batch["source_turns"]),
                      "returned_record": snapshot,
                      "fact_support_status": outcome_event["fact_support_status"]})
    assert "gold question" not in str(events) and "gold answer" not in str(events)
    result = {
        "scope": "两场真实中文来源 + 注入的上一阶段模型事实文本；Recorder 验证可选返回正文快照",
        "source_packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "injected_fact_run_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "unchanged_backend_requests": True,
        "input_outcome_pairs": pairs,
        "limits": ["注入事实取自上一阶段新 JSON 提取 POC，并非历史 Mem0.add 实际返回。",
                   "事件保存正文和指纹只保证事后可见当次返回文本，尚无逐子句来源语义支持。",
                   "调用者仍需持久化 sink；默认评测运行没有自动开启。"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pairs": len(pairs), "unchanged_backend_requests": True,
                      "frozen_returned_texts": len(pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
