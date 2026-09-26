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
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 关 mem0 PostHog 噪声，保 stdout 干净


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="配置轨道（中文：config/locomo_zh.yaml；缺省 config/default.yaml 英文轨道）")
    ap.add_argument("--prefix", default="full", help="user_id 前缀（中文轨道 zhfull）")
    ap.add_argument("--conv", default="conv-42")
    ap.add_argument("--model", default=None,
                    help="覆盖提取模型（如 glm-5.3 非 flash 档）；缺省用 config 的 LLM 模型")
    args = ap.parse_args()

    from schema_rsi.benchmarks.locomo import LocomoDataset
    from schema_rsi.config import get_settings
    from schema_rsi.evaluation.pipeline import EvaluationPipeline
    from schema_rsi.memory import Mem0Backend
    from schema_rsi.memory.mem0_backend import build_mem0_config

    settings = get_settings(args.config)
    cfg = build_mem0_config(settings)
    if args.model:
        cfg["llm"]["config"]["model"] = args.model  # 同 reingest_nonflash.py:38
        print(f"提取模型：{args.model}（覆盖 config 的 {settings.llm.model}）")
    backend = Mem0Backend(config=cfg)
    pipeline = EvaluationPipeline(backend=backend, settings=settings)

    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    case = next((c for c in ds.cases if c.metadata.get("conversation_id") == args.conv), None)
    if case is None:
        raise SystemExit(f"[FATAL] 对话 {args.conv} 不在数据集 {settings.locomo_path}")

    uid = f"{args.prefix}:locomo:{args.conv}"
    existing = backend.get_all_memories(user_id=uid)
    if existing:  # 幂等：断点重跑安全
        print(f"{args.conv}: 已有 {len(existing)} 条记忆（user_id={uid}），跳过")
        return 0

    t0 = time.perf_counter()
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
