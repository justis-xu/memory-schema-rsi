#!/usr/bin/env python3
"""Replay one fixed Chinese history with distinct run/attempt trace identities."""

import copy
import hashlib
import json
from pathlib import Path

from schema_rsi.benchmarks.longmemeval import _parse_item
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from m2_sample_lme_update_operations import ROOT

OUT = ROOT / "results/analysis/m2_trace_run_identity_20260929.json"
QUESTION_ID = "031748ae"
SOURCE = ROOT.parent / "memory-prompt/eval-datasets/longmemeval-zh/longmemeval_s_cleaned_zh.json"
OLD_SOURCE_SHA256 = "e93bd28549d25ba4092034bfe336a92f5fbce7e17693c5672a5f9d3be0d96f1f"
OLD_TRACE = ROOT / "results/analysis/m2_ingest_source_trace_20260928.json"


class Recorder:
    def __init__(self):
        self.calls = []

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append(copy.deepcopy((user_id, messages, metadata)))
        return []


def replay(case, run_id, attempt_id):
    backend, events = Recorder(), []
    pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
    pipeline.backend = backend
    count = pipeline.ingest_case(case, user_id="offline:" + case.case_id,
                                 source_trace_sink=events.append,
                                 trace_run_id=run_id, trace_attempt_id=attempt_id)
    assert count == 0
    assert len(events) == 2 * len(backend.calls)
    for input_event, outcome_event in zip(events[::2], events[1::2]):
        assert input_event["event"] == "batch_input"
        assert outcome_event["event"] == "batch_outcome"
        assert (input_event["run_id"], input_event["attempt_id"]) == (run_id, attempt_id)
        assert (outcome_event["run_id"], outcome_event["attempt_id"]) == (run_id, attempt_id)
        assert input_event["batch"]["batch_id"] == outcome_event["batch_id"]
    return backend.calls, [event["batch"]["batch_id"] for event in events[::2]]


def main():
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")
    source_bytes = SOURCE.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    rows = json.loads(source_bytes)
    index, row = next((i, r) for i, r in enumerate(rows) if r["question_id"] == QUESTION_ID)
    case = _parse_item(row, index)
    identities = [("run-a", "first"), ("run-a", "retry"), ("run-b", "first")]
    replays = [replay(case, *identity) for identity in identities]
    assert all(calls == replays[0][0] for calls, _ in replays)
    assert all(ids == replays[0][1] for _, ids in replays)
    old_trace = json.loads(OLD_TRACE.read_text())
    old_case = next(r for r in old_trace["longmemeval"] if r["case_id"] == QUESTION_ID)
    old_batch_ids_match = replays[0][1] == old_case["batch_ids"]
    assert len(replays[0][1]) == len(old_case["batch_ids"])
    mismatches = [{"call_index": i, "session_id": case.history[i]["session_id"],
                   "old_batch_id": old_id, "current_batch_id": new_id}
                  for i, (new_id, old_id) in enumerate(zip(replays[0][1], old_case["batch_ids"]))
                  if new_id != old_id]
    old_ingest = json.loads((ROOT / "results/analysis/m2_lme_ingest_order_20260928.json").read_text())
    old_probe = next(r for r in old_ingest["five_ingest_probes"] if r["question_id"] == QUESTION_ID)
    answer_source_indices = [r["call_index"] for r in old_probe["answer_source_call_order"]]
    answer_source_batches_match = all(replays[0][1][i] == old_case["batch_ids"][i]
                                      for i in answer_source_indices)
    result = {
        "scope": "固定中文 LongMemEval 工程师人数题的三次 Recorder 回放；无模型和库图写入",
        "question_id": QUESTION_ID,
        "source_sha256": source_sha256,
        "old_source_sha256": OLD_SOURCE_SHA256,
        "source_file_sha_changed": source_sha256 != OLD_SOURCE_SHA256,
        "old_ordered_batch_ids_match": old_batch_ids_match,
        "old_matching_batch_count": len(replays[0][1]) - len(mismatches),
        "old_mismatched_batches": mismatches,
        "answer_source_call_indices": answer_source_indices,
        "answer_source_batches_match": answer_source_batches_match,
        "identities": [{"run_id": r, "attempt_id": a} for r, a in identities],
        "batch_count_per_replay": len(replays[0][0]),
        "all_backend_requests_identical": True,
        "all_ordered_batch_ids_identical": True,
        "all_input_outcome_events_paired": True,
        "ordered_batch_ids_sha256": hashlib.sha256(json.dumps(replays[0][1]).encode()).hexdigest(),
        "limits": ["Recorder 没有实际提取输出；身份字段由调用者提供，现有运行脚本未启用。",
                   "相同 run/attempt 标签可被调用者重复使用，接口本身不保证全局唯一。",
                   "当前中文 HF 整文件指纹与旧报告不同；固定题是否变化以批次 ID 对账为准。",
                   "合法批次与运行身份不构成逐事实来源语义支持或答题收益。"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"question_id": QUESTION_ID,
                      "batch_count_per_replay": result["batch_count_per_replay"],
                      "requests_and_batch_ids_unchanged": True,
                      "old_ordered_batch_ids_match": old_batch_ids_match,
                      "old_mismatch_count": len(mismatches),
                      "answer_source_batches_match": answer_source_batches_match}, ensure_ascii=False))


if __name__ == "__main__":
    main()
