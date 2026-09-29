#!/usr/bin/env python3
"""Compare five archived Chinese LME source traces with the current local file."""

import copy
import difflib
import hashlib
import json
from datetime import datetime, timezone

from schema_rsi.benchmarks.longmemeval import _parse_item
from schema_rsi.evaluation.pipeline import EvaluationPipeline
from m2_verify_trace_run_identity import ROOT, SOURCE, OLD_TRACE, OLD_SOURCE_SHA256

OUT = ROOT / "results/analysis/m2_lme_source_version_drift_20260929.json"
ORDER = ROOT / "results/analysis/m2_lme_ingest_order_20260928.json"
CLAIMS = ROOT / "results/analysis/m2_lme_clause_support_20260928.json"
OLD_PACKET = ROOT / "results/analysis/m2_lme_update_operations_20260928.json"
PRIOR_CURRENT = ROOT / "results/analysis/m2_trace_run_identity_20260929.json"


class Recorder:
    def __init__(self):
        self.calls = []

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append(copy.deepcopy((user_id, messages, metadata)))
        return []


def message_sha(messages):
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def audit_case(case, old_trace, old_order, old_claims, old_packet):
    backend, events = Recorder(), []
    pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
    pipeline.backend = backend
    pipeline.ingest_case(case, user_id="offline:" + case.case_id, source_trace_sink=events.append)
    batches = [event["batch"] for event in events if event["event"] == "batch_input"]
    assert len(batches) == len(backend.calls) == len(old_trace["batch_ids"])
    assert len(batches) == len(old_order["calls"])
    changes = []
    for index, (batch, call, old_id, old_call) in enumerate(zip(
            batches, backend.calls, old_trace["batch_ids"], old_order["calls"])):
        assert batch["session_id"] == old_call["session_id"]
        assert batch["session_date"] == old_call["date"]
        assert len(call[1]) == old_call["message_count"]
        if batch["batch_id"] != old_id:
            current_message_sha = message_sha(call[1])
            changes.append({
                "call_index": index,
                "session_id": batch["session_id"],
                "old_batch_id": old_id,
                "current_batch_id": batch["batch_id"],
                "old_message_sha256": old_call["ordered_message_sha256"],
                "current_message_sha256": current_message_sha,
                "message_payload_changed": current_message_sha != old_call["ordered_message_sha256"],
                "session_id_date_and_message_count_unchanged": True,
            })
    answer_indices = [row["call_index"] for row in old_order["answer_source_call_order"]]
    answer_changes = [row for row in changes if row["call_index"] in answer_indices]
    current_sessions = {session["session_id"]: session for session in case.history}
    answer_turn_changes = []
    for archived in old_packet["source"]["chinese"]["answer_sessions"]:
        current = current_sessions[archived["session_id"]]
        assert len(archived["turns"]) == len(current["turns"])
        for index, (old_turn, new_turn) in enumerate(zip(archived["turns"], current["turns"])):
            if old_turn["role"] != new_turn["role"] or old_turn["content"] != new_turn["content"]:
                edits = [{"op": op, "old": old_turn["content"][a:b],
                          "current": new_turn["content"][c:d]}
                         for op, a, b, c, d in difflib.SequenceMatcher(
                             None, old_turn["content"], new_turn["content"],
                             autojunk=False).get_opcodes() if op != "equal"]
                answer_turn_changes.append({
                    "session_id": archived["session_id"], "source_ref": f"turn:{index}",
                    "old_role": old_turn["role"], "current_role": new_turn["role"],
                    "old_text_sha256": hashlib.sha256(old_turn["content"].encode()).hexdigest(),
                    "current_text_sha256": hashlib.sha256(new_turn["content"].encode()).hexdigest(),
                    "edits": edits,
                })
    batch_by_session = {batch["session_id"]: batch for batch in batches}
    witness_checks = []
    for claim in old_claims["claims"]:
        for witness in claim["witnesses"]:
            batch = batch_by_session[witness["session_id"]]
            source = next(s for s in batch["source_turns"]
                          if s["source_ref"] == witness["source_ref"])
            witness_checks.append({"claim_id": claim["claim_id"],
                                   "session_id": witness["session_id"],
                                   "source_ref": witness["source_ref"],
                                   "role_match": source["role"] == witness["role"],
                                   "text_match": source["content"] == witness["text"]})
    return {"question_id": case.case_id,
            "batch_count": len(batches),
            "matching_batch_count": len(batches) - len(changes),
            "changed_batches": changes,
            "answer_source_call_indices": answer_indices,
            "answer_source_batches_changed": answer_changes,
            "answer_source_turn_changes": answer_turn_changes,
            "witness_checks": witness_checks}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")
    source_bytes = SOURCE.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source_rows = json.loads(source_bytes)
    old_trace = json.loads(OLD_TRACE.read_text())
    old_order = json.loads(ORDER.read_text())
    old_claims = json.loads(CLAIMS.read_text())
    old_packet = json.loads(OLD_PACKET.read_text())
    prior_current = json.loads(PRIOR_CURRENT.read_text())
    trace_by_id = {row["case_id"]: row for row in old_trace["longmemeval"]}
    order_by_id = {row["question_id"]: row for row in old_order["five_ingest_probes"]}
    claims_by_id = {row["question_id"]: row for row in old_claims["cases"]}
    packet_by_id = {row["question_id"]: row for row in old_packet["cases"]}
    assert set(trace_by_id) == set(order_by_id) == set(claims_by_id) == set(packet_by_id)
    cases = []
    for index, row in enumerate(source_rows):
        question_id = row["question_id"]
        if question_id in trace_by_id:
            case = _parse_item(row, index)
            cases.append(audit_case(case, trace_by_id[question_id],
                                    order_by_id[question_id], claims_by_id[question_id],
                                    packet_by_id[question_id]))
    assert len(cases) == 5
    changed = sum(len(row["changed_batches"]) for row in cases)
    witnesses = [check for row in cases for check in row["witness_checks"]]
    result = {
        "scope": "五道固定中文 LongMemEval 知识更新题；当前文件与 2026-09-28 归档输入逐批对照",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_source_sha256": source_sha,
        "prior_current_source_sha256": prior_current["source_sha256"],
        "old_source_sha256": OLD_SOURCE_SHA256,
        "source_file_sha_changed": source_sha != OLD_SOURCE_SHA256,
        "summary": {"case_count": len(cases),
                    "batch_count": sum(row["batch_count"] for row in cases),
                    "changed_batch_count": changed,
                    "changed_answer_source_batch_count": sum(len(row["answer_source_batches_changed"])
                                                             for row in cases),
                    "changed_answer_source_turn_count": sum(len(row["answer_source_turn_changes"])
                                                            for row in cases),
                    "witness_count": len(witnesses),
                    "witness_role_text_match_count": sum(w["role_match"] and w["text_match"]
                                                         for w in witnesses)},
        "cases": cases,
        "limits": ["旧完整 HF 文件未保存；旧答案来源场次的逐轮正文另有归档，可定位这些场次的变更轮次。其他变更批次不能还原旧正文或原因。",
                   "当前中文文件在本轮与上一轮间继续变化，本结果严格对应 current_source_sha256 所标识的读取快照。",
                   "只比较既有五题和其已审来源 turn，不估全 470 题或中文 Graph/Jev/RSI 净收益。",
                   "Recorder 返回空输出；没有真实提取、逐事实自动对齐、答题或库图写入。"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
