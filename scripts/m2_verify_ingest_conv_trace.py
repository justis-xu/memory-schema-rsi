#!/usr/bin/env python3
"""Exercise the opt-in CLI trace writer over two archived Chinese sessions."""

import copy
import hashlib
import json
import tempfile
from pathlib import Path

from ingest_conv import file_sha256, ingest_with_trace
from schema_rsi.benchmarks.base import BenchmarkCase
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from schema_rsi.memory.base import MemoryRecord
from m2_poc_lme_update_extraction import ROOT, PACKET

OUT = ROOT / "results/analysis/m2_ingest_conv_trace_20260929.json"
RUN = ROOT / "results/analysis/m2_lme_update_extraction_20260929.json"
CONFIG = ROOT / "config/locomo_zh.yaml"


class Recorder:
    def __init__(self, by_session):
        self.by_session = by_session
        self.calls = []

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append(copy.deepcopy((user_id, messages, metadata)))
        return [copy.deepcopy(self.by_session[metadata["session_id"]])]


def pipeline(backend):
    result = EvaluationPipeline.__new__(EvaluationPipeline)
    result.backend = backend
    return result


def main():
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")
    packet_bytes = PACKET.read_bytes()
    packet = json.loads(packet_bytes)
    archived = next(c for c in packet["cases"] if c["question_id"] == "031748ae")
    source_sessions = archived["source"]["chinese"]["answer_sessions"]
    history = [{"session_id": session["session_id"], "date": session["date"],
                "turns": [{"role": turn["role"], "content": turn["content"]}
                          for turn in session["turns"]]}
               for session in source_sessions]
    case = BenchmarkCase("031748ae", "longmemeval_s", history,
                         "gold question sentinel", "gold answer sentinel")
    run = json.loads(RUN.read_text())
    policy = next(c for c in run["calls"] if c["question_id"] == case.case_id
                  and c["arm"] == "source_policy")
    facts = policy["parsed_response"]["facts"]
    chosen = [facts[1], facts[5]]
    outputs = {}
    for session, fact in zip(source_sessions, chosen):
        sid = session["session_id"]
        outputs[sid] = MemoryRecord("offline:" + sid,
                                    f"{fact['subject']} {fact['relation']} {fact['object']}",
                                    {"session_id": sid, "session_date": session["date"],
                                     "event": "ADD"})
    plain, traced = Recorder(outputs), Recorder(outputs)
    assert pipeline(plain).ingest_case(case, user_id="offline:" + case.case_id) == 2
    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "trace.jsonl"
        assert ingest_with_trace(pipeline(traced), case, "offline:" + case.case_id,
                                 trace_path=trace_path, run_id="fixed-offline-run",
                                 attempt_id="first", source_path=PACKET,
                                 source_sha256=hashlib.sha256(packet_bytes).hexdigest(),
                                 config_sha256=file_sha256(CONFIG),
                                 effective_config={"llm_model": "offline-recorder",
                                                   "source_prompt": "archived-model-facts"}) == 2
        trace_bytes = trace_path.read_bytes()
    assert plain.calls == traced.calls
    events = [json.loads(line) for line in trace_bytes.decode().splitlines()]
    assert [e["event"] for e in events] == ["run_started", "batch_input", "batch_outcome",
                                                "batch_input", "batch_outcome", "run_completed"]
    assert events[0]["source_file_sha256"] == events[-1]["source_file_sha256_after"]
    assert events[-1]["source_file_unchanged"] is True
    for before, after in ((events[1], events[2]), (events[3], events[4])):
        assert before["batch"]["batch_id"] == after["batch_id"]
        sid = before["batch"]["session_id"]
        assert after["returned_records"][0]["content"] == outputs[sid].content
    assert "gold question sentinel" not in str(events)
    assert "gold answer sentinel" not in str(events)
    result = {
        "scope": "两场归档中文原话的单对话 CLI 可选 JSONL 来源账离线回放",
        "source_packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "config_file_sha256": file_sha256(CONFIG),
        "trace_file_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "event_types": [e["event"] for e in events],
        "backend_requests_unchanged": True,
        "batch_ids": [events[1]["batch"]["batch_id"], events[3]["batch"]["batch_id"]],
        "returned_content_sha256": [events[2]["returned_records"][0]["content_sha256"],
                                    events[4]["returned_records"][0]["content_sha256"]],
        "source_file_unchanged": True,
        "limits": ["Recorder 注入上一阶段 POC 的中文事实，不是历史 Mem0 实际提取。",
                   "临时 JSONL 仅用于离线验证，机器结果留存其指纹和事件结构。",
                   "入口只加了显式单对话留痕选项；全量评测脚本仍未开启。"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"events": len(events), "batch_pairs": 2,
                      "backend_requests_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
