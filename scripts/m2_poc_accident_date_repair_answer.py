"""Bounded frozen-context answer POC for one source-misdated memory.

Calls only the answer model, never the judge, retriever, vector store or graph.
"""

import hashlib
import json
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import build_official_answer_prompt
from schema_rsi.llm.chat import make_chat_client
from schema_rsi.memory.base import MemoryRecord


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results_zh/zhfull_fused_locomo_20260925_161049.jsonl"
SOURCE = ROOT / "results/analysis/m2_output_origin_sample_20260928.json"
OUT = ROOT / "results/analysis/m2_accident_date_repair_answer_20260929.json"
TARGET = "f87df58f-81e4-46e4-a2bc-22e62b98c625"
CASES = ("locomo_conv-26_qa74", "locomo_conv-26_qa144")
ORDER = ("original", "date_repaired", "date_repaired", "original")
OLD = "在2023年10月20日的公路旅行中"
NEW = "在2023年10月20日谈话前的上周末公路旅行中"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_record(item):
    return MemoryRecord(id=item["id"], content=item["content"], metadata=dict(item["metadata"]))


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite prior calls: {OUT}")
    source = json.loads(SOURCE.read_text())
    audited, = [item for item in source["items"] if item["memory"]["embedding_id"] == TARGET]
    assert audited["memory"]["content"].count(OLD) == 1
    assert audited["supported_origin_session"] == "session_18"
    assert audited["memory"]["session_id"] == "session_19"
    rows = {row["case_id"]: row for row in map(json.loads, RUN.open()) if row["case_id"] in CASES}
    assert set(rows) == set(CASES)
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    assert settings.llm.model == "glm-5.3-flash"
    prepared = {}
    result = {
        "scope": "Two source-selected Chinese LoCoMo cases from one archived fixed 15-slot context; only the unsupported date phrase in one memory is repaired. ABBA per case; zero judge calls.",
        "source_run": str(RUN.relative_to(ROOT)), "source_run_sha256": sha(RUN.read_bytes()),
        "source_audit": str(SOURCE.relative_to(ROOT)), "source_audit_sha256": sha(SOURCE.read_bytes()),
        "model": settings.llm.model, "temperature": 0.0, "max_answer_calls": 8,
        "max_completion_tokens_per_call": 1500, "order_per_case": ORDER,
        "target_memory_id": TARGET, "original_phrase": OLD, "repaired_phrase": NEW,
        "cases": [], "calls_sent": 0,
        "limits": [
            "Old archived context; current pure-Chinese memory-store and retrieval quality are not tested.",
            "Source repair changes the one memory body only; its session_19 metadata remains source-misaligned.",
            "Two fixed questions and two answers per arm cannot estimate a population effect or noise distribution.",
            "A response difference cannot by itself establish that this one memory caused the historical answer.",
        ],
    }
    for case_id in CASES:
        row = rows[case_id]
        ids = row["metadata"]["answer_context_ids"]
        assert len(ids) == 15 and ids.count(TARGET) == 1
        candidates = {m["id"]: m for m in row["retrieved_memories"] + row["graph_memories"]}
        assert all(i in candidates for i in ids)
        base = [to_record(candidates[i]) for i in ids]
        target, = [m for m in base if m.id == TARGET]
        assert target.content == audited["memory"]["content"]
        fixed = [MemoryRecord(id=m.id, content=m.content.replace(OLD, NEW, 1), metadata=dict(m.metadata))
                 if m.id == TARGET else m for m in base]
        assert sum(a.content != b.content for a, b in zip(base, fixed)) == 1
        assert all(a.id == b.id and a.metadata == b.metadata for a, b in zip(base, fixed))
        contexts = {"original": base, "date_repaired": fixed}
        prompts = {arm: build_official_answer_prompt(row["question"], ctx, "2023")
                   for arm, ctx in contexts.items()}
        prepared[case_id] = contexts
        result["cases"].append({
            "case_id": case_id, "question": row["question"],
            "gold_for_posthoc_review_only": row["expected_answer"],
            "target_rank": ids.index(TARGET) + 1, "context_ids": ids,
            "prompt_sha256": {k: sha(v.encode()) for k, v in prompts.items()},
            "original_target_text": target.content,
            "repaired_target_text": fixed[ids.index(TARGET)].content,
            "calls": [],
        })
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    for case in result["cases"]:
        for arm in ORDER:
            assert result["calls_sent"] < result["max_answer_calls"]
            result["calls_sent"] += 1
            try:
                answer, info = answerer.answer(case["question"], prepared[case["case_id"]][arm],
                                               temperature=0.0, reference_date="2023", benchmark="locomo")
                if not answer:
                    raise RuntimeError("empty answer")
                call = {"index": result["calls_sent"], "arm": arm, "answer": answer, "usage": info["usage"]}
            except Exception as exc:
                call = {"index": result["calls_sent"], "arm": arm,
                        "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)}
                case["calls"].append(call)
                result["stopped_after_failure"] = True
                OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                print(case["case_id"], call, flush=True)
                return
            case["calls"].append(call)
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(case["case_id"], arm, answer[:130].replace("\n", " "), flush=True)
    assert result["calls_sent"] == 8
    result["stopped_after_failure"] = False
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
