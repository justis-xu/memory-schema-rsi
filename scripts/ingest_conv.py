#!/usr/bin/env python
"""单对话中文提取：按 --config 轨道对 --conv 做与 full run 完全一致的 ingest。

与 scripts/eval_full.py 的 ingest 阶段唯一差别：只做一个 conversation。
模型/向量库/集合名全部来自同一 config（中文轨道 config/locomo_zh.yaml：
collection schema_rsi_memories_zh、data/vector_store_zh/），提取窗口为全量
（max_turns=None）、日期锚注入等行为与 full run 逐字一致（同一
EvaluationPipeline.ingest_case 代码路径，见 pipeline.py:50）。
user_id = f"{prefix}:locomo:{conv}"；已有记忆则跳过（幂等，断点重跑安全）。
--model 不传用 config 的 LLM 模型（flash 档）；传 --model glm-5.3 时覆盖
（做法同 scripts/reingest_nonflash.py：build_mem0_config 后改 llm.config.model，
其余不动）。

用法：
  .venv/bin/python scripts/ingest_conv.py --config config/locomo_zh.yaml \
      --prefix zhfull --conv conv-42
  .venv/bin/python scripts/ingest_conv.py --config config/locomo_zh.yaml \
      --prefix zhnf53 --conv conv-42 --model glm-5.3   # 非 flash 对照库
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 关 mem0 PostHog 噪声，保 stdout 干净


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ingest_config(cfg: dict) -> dict:
    """Record effective extraction settings without credentials or raw endpoints."""
    llm = cfg['llm']['config']
    embedder = cfg['embedder']['config']
    vector = cfg['vector_store']['config']
    custom = cfg.get('custom_instructions') or ''
    return {
        'llm_model': llm['model'],
        'llm_temperature': llm['temperature'],
        'llm_max_tokens': llm['max_tokens'],
        'disable_thinking': llm['disable_thinking'],
        'llm_base_url_sha256': hashlib.sha256(str(llm.get('openai_base_url', '')).encode()).hexdigest(),
        'custom_instructions_sha256': hashlib.sha256(custom.encode()).hexdigest(),
        'embedding_model': embedder['model'],
        'embedding_base_url_sha256': hashlib.sha256(str(embedder.get('openai_base_url', '')).encode()).hexdigest(),
        'collection_name': vector['collection_name'],
    }


def ingest_with_trace(pipeline, case, uid: str, *, trace_path: Path,
                      run_id: str, attempt_id: str, source_path: Path,
                      source_sha256: str, config_sha256: str,
                      effective_config: dict) -> int:
    """Run one conversation with an exclusive source/output trace file."""
    from schema_rsi.evaluation.source_trace import JsonlTraceSink

    with JsonlTraceSink(trace_path) as sink:
        sink({'event': 'run_started', 'run_id': run_id, 'attempt_id': attempt_id,
              'case_id': case.case_id, 'user_id': uid,
              'source_file_sha256': source_sha256,
              'config_file_sha256': config_sha256,
              'effective_config': effective_config,
              'effective_config_sha256': hashlib.sha256(json.dumps(
                  effective_config, ensure_ascii=False, sort_keys=True,
                  separators=(',', ':')).encode()).hexdigest()})
        added = pipeline.ingest_case(case, max_turns=None, user_id=uid,
                                     source_trace_sink=sink,
                                     trace_run_id=run_id, trace_attempt_id=attempt_id)
        source_sha_after = file_sha256(source_path)
        sink({'event': 'run_completed', 'run_id': run_id, 'attempt_id': attempt_id,
              'case_id': case.case_id, 'added_record_count': added,
              'source_file_sha256_after': source_sha_after,
              'source_file_unchanged': source_sha_after == source_sha256,
              'ingest_error_count': len(getattr(pipeline, 'ingest_errors', []))})
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="配置轨道（中文：config/locomo_zh.yaml；缺省 config/default.yaml 英文轨道）")
    ap.add_argument("--prefix", default="full", help="user_id 前缀（中文轨道 zhfull）")
    ap.add_argument("--conv", default="conv-42")
    ap.add_argument("--model", default=None,
                    help="覆盖提取模型（如 glm-5.3 非 flash 档）；缺省用 config 的 LLM 模型")
    ap.add_argument("--trace-output", default=None,
                    help="可选：独占创建来源账 JSONL；已有文件拒绝覆盖")
    ap.add_argument("--trace-run-id", default=None,
                    help="可选：本次来源账运行 ID；未传则自动生成 UUID")
    ap.add_argument("--trace-attempt-id", default="cli-1",
                    help="可选：本次 CLI 尝试 ID；默认 cli-1")
    args = ap.parse_args()
    if (args.trace_run_id is not None or args.trace_attempt_id != 'cli-1') and not args.trace_output:
        ap.error("--trace-run-id/--trace-attempt-id 需要 --trace-output")

    from schema_rsi.benchmarks.locomo import LocomoDataset
    from schema_rsi.config import get_settings
    from schema_rsi.evaluation.pipeline import EvaluationPipeline
    from schema_rsi.memory import Mem0Backend
    from schema_rsi.memory.mem0_backend import build_mem0_config

    settings = get_settings(args.config)
    source_path = settings.locomo_path
    source_sha_before = file_sha256(source_path) if args.trace_output else None
    config_path = Path(args.config or 'config/default.yaml')
    config_sha = file_sha256(config_path) if args.trace_output else None
    cfg = build_mem0_config(settings)
    if args.model:
        cfg["llm"]["config"]["model"] = args.model  # 同 reingest_nonflash.py:38
        print(f"提取模型：{args.model}（覆盖 config 的 {settings.llm.model}）")
    backend = Mem0Backend(config=cfg)
    pipeline = EvaluationPipeline(backend=backend, settings=settings)

    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    if args.trace_output and file_sha256(source_path) != source_sha_before:
        raise SystemExit("[FATAL] 数据文件在加载期间变化，未开始写入")
    case = next((c for c in ds.cases if c.metadata.get("conversation_id") == args.conv), None)
    if case is None:
        raise SystemExit(f"[FATAL] 对话 {args.conv} 不在数据集 {settings.locomo_path}")

    uid = f"{args.prefix}:locomo:{args.conv}"
    existing = backend.get_all_memories(user_id=uid)
    if existing:  # 幂等：断点重跑安全
        print(f"{args.conv}: 已有 {len(existing)} 条记忆（user_id={uid}），跳过")
        return 0

    t0 = time.perf_counter()
    if args.trace_output:
        run_id = args.trace_run_id or uuid.uuid4().hex
        added = ingest_with_trace(
            pipeline, case, uid, trace_path=Path(args.trace_output),
            run_id=run_id, attempt_id=args.trace_attempt_id,
            source_path=source_path, source_sha256=source_sha_before,
            config_sha256=config_sha, effective_config=safe_ingest_config(cfg))
        print(f"来源账：{args.trace_output}（run_id={run_id}）")
    else:
        added = pipeline.ingest_case(case, max_turns=None, user_id=uid)  # 全量窗口
    n = len(backend.get_all_memories(user_id=uid))
    errs = getattr(pipeline, "ingest_errors", [])
    print(f"{args.conv}: ingest +{added} 条记忆 → 库存 {n} 条"
          f"（{time.perf_counter() - t0:.0f}s, sessions={len(case.history)}, "
          f"turns={case.history_stats()['turns']}, ingest_errors={len(errs)}）")
    for e in errs:
        print(f"  ingest_error {e['session_id']}: {e['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
