#!/usr/bin/env python3
"""One bounded real Mem0 ingest on an archived Chinese source session."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/analysis/m2_lme_update_operations_20260928.json"
OUT = ROOT / "results/analysis/m2_isolated_mem0_ingest_20260929.json"
TRACE = ROOT / "results/analysis/m2_isolated_mem0_ingest_trace_20260929.jsonl"
QUESTION_ID = "031748ae"
SESSION_ID = "answer_8748f791_1"
MAX_EXTRACTION_CALLS = 1
MAX_EMBEDDING_HTTP_CALLS = 6


def save(report):
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def frozen_session():
    packet_bytes = PACKET.read_bytes()
    packet = json.loads(packet_bytes)
    case = next(c for c in packet["cases"] if c["question_id"] == QUESTION_ID)
    session = next(s for s in case["source"]["chinese"]["answer_sessions"]
                   if s["session_id"] == SESSION_ID)
    return packet_bytes, session


def prepare():
    if OUT.exists() or TRACE.exists():
        raise SystemExit("refusing overwrite of prepared result or trace")
    from schema_rsi.config import get_settings

    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    packet_bytes, session = frozen_session()
    report = {
        "scope": "隔离临时库、固定中文 LongMemEval 工程师早场真实 Mem0 V3 写入",
        "status": "prepared",
        "source_packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "question_id": QUESTION_ID, "session_id": SESSION_ID,
        "session_date": session["date"],
        "source_turn_count": len(session["turns"]),
        "llm_model": settings.llm.model,
        "embedding_model": settings.embedding.model,
        "max_extraction_calls": MAX_EXTRACTION_CALLS,
        "max_embedding_http_calls": MAX_EMBEDDING_HTTP_CALLS,
        "sdk_max_retries": 0,
        "answer_or_judge_calls": 0,
        "trace_path": str(TRACE.relative_to(ROOT)),
        "llm_calls": [], "embedding_calls": [],
        "limits": ["不是历史 Mem0 写入重放：当前模型、提示和代码版本与旧运行可不同。",
                   "仅一场旧源，不测跨场更新或网球频率；临时库退出即销毁。",
                   "返回正文和库存仍需独立人工逐事实回源，不自动代表安全记忆。"],
    }
    save(report)
    print("prepared one archived Chinese session; zero model calls")


class EmbeddingHTTPProxy:
    def __init__(self, endpoint, report):
        self.endpoint = endpoint
        self.report = report

    def create(self, **kwargs):
        calls = self.report["embedding_calls"]
        if len(calls) >= MAX_EMBEDDING_HTTP_CALLS:
            raise RuntimeError("embedding_http_call_limit")
        payload = kwargs.get("input", [])
        calls.append({"call_index": len(calls) + 1, "model": kwargs.get("model"),
                      "input_count": len(payload),
                      "input_sha256": hashlib.sha256(json.dumps(
                          payload, ensure_ascii=False).encode()).hexdigest()})
        save(self.report)
        return self.endpoint.create(**kwargs)


class ClientProxy:
    def __init__(self, client, report):
        self.client = client
        self.embeddings = EmbeddingHTTPProxy(client.embeddings, report)

    def __getattr__(self, name):
        return getattr(self.client, name)


def run():
    report = json.loads(OUT.read_text())
    if report["status"] != "prepared" or TRACE.exists():
        raise SystemExit("refusing to rerun a started diagnostic")
    packet_bytes, session = frozen_session()
    assert hashlib.sha256(packet_bytes).hexdigest() == report["source_packet_sha256"]

    from schema_rsi.benchmarks.base import BenchmarkCase
    from schema_rsi.config import get_settings
    from schema_rsi.evaluation.pipeline import EvaluationPipeline
    from schema_rsi.evaluation.source_trace import JsonlTraceSink
    from schema_rsi.memory.mem0_backend import Mem0Backend, build_mem0_config

    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    assert settings.llm.model == report["llm_model"]
    assert settings.embedding.model == report["embedding_model"]
    history = [{"session_id": SESSION_ID, "date": session["date"],
                "turns": [{"role": t["role"], "content": t["content"]}
                          for t in session["turns"]]}]
    case = BenchmarkCase(QUESTION_ID, "longmemeval_s", history,
                         "diagnostic question excluded from extraction",
                         "diagnostic gold excluded from extraction")
    report["status"] = "running"
    save(report)
    with tempfile.TemporaryDirectory(prefix="m2-mem0-isolated-") as directory:
        cfg = build_mem0_config(settings)
        cfg["vector_store"]["config"]["path"] = str(Path(directory) / "chroma")
        cfg["vector_store"]["config"]["collection_name"] = "m2_isolated_engineer"
        cfg["history_db_path"] = str(Path(directory) / "history.db")
        backend = Mem0Backend(settings=settings, config=cfg)
        llm = backend.raw.llm
        llm.client = llm.client.with_options(max_retries=0, timeout=45)
        embedding = backend.raw.embedding_model
        embedding.client = ClientProxy(
            embedding.client.with_options(max_retries=0, timeout=45), report)
        original_generate = llm.generate_response

        def bounded_generate(*args, **kwargs):
            if len(report["llm_calls"]) >= MAX_EXTRACTION_CALLS:
                raise RuntimeError("extraction_call_limit")
            messages = kwargs.get("messages", args[0] if args else None)
            call = {"call_index": len(report["llm_calls"]) + 1,
                    "messages": messages,
                    "response_format": kwargs.get("response_format")}
            report["llm_calls"].append(call)
            save(report)
            response = original_generate(*args, **kwargs)
            call["raw_response"] = response
            save(report)
            return response

        llm.generate_response = bounded_generate
        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        pipeline.backend = backend
        try:
            with JsonlTraceSink(TRACE) as sink:
                sink({"event": "run_started", "run_id": "m2-isolated-engineer",
                      "source_packet_sha256": report["source_packet_sha256"],
                      "model": report["llm_model"]})
                added = pipeline.ingest_case(
                    case, user_id="m2-isolated:engineer-early",
                    source_trace_sink=sink,
                    trace_run_id="m2-isolated-engineer", trace_attempt_id="first")
                sink({"event": "run_completed", "run_id": "m2-isolated-engineer",
                      "added_record_count": added,
                      "ingest_error_count": len(getattr(pipeline, "ingest_errors", []))})
            report["returned_add_count"] = added
            report["ingest_errors"] = [
                {"session_id": e["session_id"], "error_type": "backend_error"}
                for e in getattr(pipeline, "ingest_errors", [])]
            inventory = backend.get_all_memories(user_id="m2-isolated:engineer-early")
            report["isolated_inventory"] = [
                {"id": r.id, "content": r.content,
                 "content_sha256": hashlib.sha256(r.content.encode()).hexdigest(),
                 "session_id": r.metadata.get("session_id"),
                 "session_date": r.metadata.get("session_date")}
                for r in inventory]
            report["status"] = "completed" if not report["ingest_errors"] else "backend_error"
        except Exception as exc:
            report["status"] = "failed"
            report["error_type"] = type(exc).__name__
            report["http_status"] = getattr(exc, "status_code", None)
        finally:
            if TRACE.exists():
                trace_bytes = TRACE.read_bytes()
                report["trace_sha256"] = hashlib.sha256(trace_bytes).hexdigest()
                report["trace_event_count"] = len(trace_bytes.splitlines())
            report["llm_call_count"] = len(report["llm_calls"])
            report["embedding_http_call_count"] = len(report["embedding_calls"])
            save(report)
    print(json.dumps({k: report.get(k) for k in (
        "status", "llm_call_count", "embedding_http_call_count", "returned_add_count",
        "trace_event_count")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose --prepare or --run")
    prepare() if args.prepare else run()
