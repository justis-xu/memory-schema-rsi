#!/usr/bin/env python3
"""Read-only audit of role labels across Chinese LoCoMo and LongMemEval."""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
LME_TRACE = ROOT / "results/analysis/m2_isolated_mem0_ingest_trace_20260929.jsonl"
OUT = ROOT / "results/analysis/m2_cross_benchmark_role_semantics_20260929.json"
WITNESS_ID = "7d5bc1cd-026d-4baf-b6cf-007667a72128"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit("refusing to overwrite audit")
    data = json.loads(DATASET.read_text())
    conv = next(x["conversation"] for x in data if x["sample_id"] == "conv-43")
    source = [t for t in conv["session_1"] if t["dia_id"] in ("D1:3", "D1:5", "D1:7")]
    assert conv["speaker_a"] == "蒂姆" and conv["speaker_b"] == "约翰"
    assert all(t["speaker"] == conv["speaker_b"] for t in source)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT e.embedding_id,
          MAX(CASE WHEN m.key='data' THEN m.string_value END) AS content,
          MAX(CASE WHEN m.key='user_id' THEN m.string_value END) AS owner,
          MAX(CASE WHEN m.key='session_id' THEN m.string_value END) AS session_id,
          MAX(CASE WHEN m.key='attributed_to' THEN m.string_value END) AS attributed_to
        FROM embeddings e JOIN embedding_metadata m ON e.id=m.id GROUP BY e.id
    """).fetchall()
    conn.close()
    zh = [r for r in rows if r[2] and r[2].startswith("zhfull:locomo:")]
    witness, = [r for r in zh if r[0] == WITNESS_ID]
    assert witness[2] == "zhfull:locomo:conv-43" and witness[3] == "session_1"
    assert witness[4] == "assistant" and "明尼苏达狼队" in witness[1]
    questions = next(x["qa"] for x in data if x["sample_id"] == "conv-43")
    qa = [{"qa_index": i, "question": questions[i]["question"],
           "answer": questions[i]["answer"], "evidence": questions[i]["evidence"]}
          for i in (71, 72)]
    assert qa[0]["evidence"] == ["D1:5"] and qa[1]["evidence"] == ["D1:7"]

    lme_input = next(json.loads(line)["batch"] for line in LME_TRACE.read_text().splitlines()
                     if json.loads(line)["event"] == "batch_input")
    assert lme_input["benchmark"] == "longmemeval_s"
    lme_assistant = [t for t in lme_input["source_turns"] if t["role"] == "assistant"]
    result = {
        "scope": "同一role/attributed_to字符串在两个中文基准中的来源语义；当前Chroma只读快照",
        "input_sha256": {"locomo_zh": sha(DATASET), "chroma_sqlite": sha(DB),
                         "isolated_lme_trace": sha(LME_TRACE)},
        "source_adapter_semantics": {
            "locomo": "speaker_a映射user、speaker_b映射assistant；两者均为数据集对话人物",
            "longmemeval_s": "原文件role为user/assistant；assistant是助理回复，不是另一位命名的人类speaker",
        },
        "chroma_counts": {
            "all_records": len(rows), "zhfull_locomo_records": len(zh),
            "zhfull_locomo_attributed_to": dict(Counter(r[4] or "missing" for r in zh)),
            "owner_count": len({r[2] for r in zh}),
        },
        "locomo_witness": {
            "memory_id": witness[0], "content": witness[1], "owner": witness[2],
            "session_id": witness[3], "attributed_to": witness[4],
            "speaker_a": conv["speaker_a"], "speaker_b": conv["speaker_b"],
            "source_turns": [{"dia_id": t["dia_id"], "speaker": t["speaker"],
                              "text": t["text"]} for t in source],
            "related_qa": qa,
        },
        "lme_assistant_turns_in_isolated_case": len(lme_assistant),
        "model_calls": 0,
        "limits": [
            "Chroma计数只是当前快照的模型标签分布，不是真实来源类型的人工准确率。",
            "真人LoCoMo例子证明不能跨基准统一过滤assistant；不证明该记忆是答题唯一载体。",
            "LongMemEval助理回复可能被用户后来确认；需逐子句看原话，不能只凭role整条删除。",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["chroma_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
