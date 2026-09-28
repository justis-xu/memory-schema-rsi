#!/usr/bin/env python3
"""Link reviewed source clauses to actual traced Chinese LME input batches.

Uses the frozen HF bytes and a recording backend. No extraction, model, vector
store, or graph service is called.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from schema_rsi.benchmarks.longmemeval import _parse_item
from schema_rsi.evaluation.pipeline import EvaluationPipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/xu/git/memory-prompt/eval-datasets/longmemeval-zh/longmemeval_s_cleaned_zh.json")
CLAIMS = ROOT / "results/analysis/m2_lme_clause_support_20260928.json"
TRACE = ROOT / "results/analysis/m2_ingest_source_trace_20260928.json"
OUTPUT = ROOT / "results/analysis/m2_lme_batch_claim_link_20260928.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Recorder:
    def __init__(self) -> None:
        self.calls = []

    def add_memory(self, user_id, messages, metadata=None):
        self.calls.append(copy.deepcopy((user_id, messages, metadata)))
        return []


def replay(case):
    backend = Recorder()
    events = []
    pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
    pipeline.backend = backend
    count = pipeline.ingest_case(case, user_id="offline:" + case.case_id,
                                 source_trace_sink=events.append)
    assert count == 0
    batches = [event["batch"] for event in events if event["event"] == "batch_input"]
    outcomes = [event for event in events if event["event"] == "batch_outcome"]
    assert len(backend.calls) == len(batches) == len(outcomes) == len(case.history)
    for call, batch, outcome in zip(backend.calls, batches, outcomes):
        assert batch["input_messages"] == call[1]
        assert batch["batch_id"] == outcome["batch_id"]
        assert outcome["status"] == "backend_returned"
        assert all("has_answer" not in turn for turn in batch["source_turns"])
        assert "question" not in batch and "answer" not in batch
    return batches


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing overwrite: {OUTPUT}")
    sidecar = json.loads(CLAIMS.read_text())
    archived = json.loads(TRACE.read_text())
    expected_sha = archived["input_sha256"]["lme_chinese"]
    actual_sha = sha(DATA)
    assert actual_sha == expected_sha, (actual_sha, expected_sha)
    wanted = {row["question_id"]: row for row in sidecar["cases"]}
    archived_cases = {row["case_id"]: row for row in archived["longmemeval"]}
    assert set(wanted) == set(archived_cases)

    rows = []
    for index, raw in enumerate(json.loads(DATA.read_text())):
        question_id = raw["question_id"]
        if question_id not in wanted:
            continue
        batches = replay(_parse_item(raw, index))
        assert [batch["batch_id"] for batch in batches] == archived_cases[question_id]["batch_ids"]
        by_session = {batch["session_id"]: batch for batch in batches}
        assert len(by_session) == len(batches)
        claims = []
        for atom in wanted[question_id]["claims"]:
            witnesses = []
            for witness in atom["witnesses"]:
                batch = by_session[witness["session_id"]]
                source = next(turn for turn in batch["source_turns"]
                              if turn["source_ref"] == witness["source_ref"])
                assert source["content"] == witness["text"]
                assert source["role"] == witness["role"]
                witnesses.append({
                    "session_id": witness["session_id"],
                    "batch_id": batch["batch_id"],
                    "source_ref": source["source_ref"],
                    "role": source["role"],
                    "source_content_sha256": hashlib.sha256(source["content"].encode()).hexdigest(),
                })
            claims.append({"claim_id": atom["claim_id"], "source_support": atom["source_support"],
                           "witnesses": witnesses})
        rows.append({"question_id": question_id, "history_batch_count": len(batches),
                     "claims": claims})
    assert len(rows) == len(wanted)
    report = {
        "scope": "五道固定中文 LongMemEval 题的人工诊断子句到实际 ingest_case 输入批次的合法定位",
        "dataset_path": str(DATA),
        "dataset_sha256": actual_sha,
        "source_claims": str(CLAIMS.relative_to(ROOT)),
        "source_trace": str(TRACE.relative_to(ROOT)),
        "checks": {
            "all_232_batch_ids_match_archived_trace": sum(row["history_batch_count"] for row in rows) == 232,
            "all_claim_witnesses_match_traced_input_text_and_role": True,
            "recorder_returned_no_memories": True,
            "all_history_batches_replayed_without_gold_input_fields": True,
        },
        "cases": rows,
        "limits": [
            "batch_id 是输入内容身份，不是运行/重试 ID。",
            "合法 source_ref 与正文一致仍不证明子句语义得到支持；支持强度来自人工侧车。",
            "Recorder 返回空列表，没有实际 Mem0 输出事实或自动来源映射。",
            "仅本机这一份 HF 文件通过 SHA；图、缓存、模型及当前记忆库未参与运行。",
        ],
    }
    assert all(report["checks"].values())
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "batches": sum(row["history_batch_count"] for row in rows),
                      "claims": sum(len(row["claims"]) for row in rows), "checks": report["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
