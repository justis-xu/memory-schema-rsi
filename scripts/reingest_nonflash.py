#!/usr/bin/env python
"""隔离重灌：指定 conversation 用非 flash 模型重做 Mem0 提取（对照提取档位）。

与 full run 唯二的差别：
  1. llm.model 换成 --model（默认 glm-5.3 非 flash 档）
  2. user_id 前缀换 --prefix（默认 nf53），与主库 full:locomo:* 完全隔离，
     不触碰 flash 主库的任何数据

用法：
  .venv/bin/python scripts/reingest_nonflash.py --conv conv-42 --conv conv-43
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--conv", action="append", required=True)
    p.add_argument("--model", default="glm-5.3")
    p.add_argument("--prefix", default="nf53")
    args = p.parse_args()

    from schema_rsi.benchmarks.locomo import LocomoDataset
    from schema_rsi.config import get_settings
    from schema_rsi.evaluation.pipeline import EvaluationPipeline
    from schema_rsi.memory.mem0_backend import Mem0Backend, build_mem0_config

    settings = get_settings()
    cfg = build_mem0_config(settings)
    cfg["llm"]["config"]["model"] = args.model
    print(f"提取模型：{args.model}（主库为 {settings.llm.model}）")
    backend = Mem0Backend(config=cfg)
    pipeline = EvaluationPipeline(backend=backend, settings=settings)

    ds = LocomoDataset()
    ds.load()
    summary = {}
    for conv in args.conv:
        uid = f"{args.prefix}:locomo:{conv}"
        existing = backend.get_all_memories(user_id=uid)
        if existing:
            print(f"  {conv}: 已有 {len(existing)} 条，跳过")
            summary[conv] = {"memories": len(existing), "skipped": True}
            continue
        case = next(c for c in ds.cases if c.metadata.get("conversation_id") == conv)
        t0 = time.perf_counter()
        added = pipeline.ingest_case(case, max_turns=None, user_id=uid)
        n = len(backend.get_all_memories(user_id=uid))
        dt = time.perf_counter() - t0
        print(f"  {conv}: ingest +{added} → 库存 {n} 条（{dt:.0f}s，"
              f"ingest_errors={len(getattr(pipeline, 'ingest_errors', []))}）")
        summary[conv] = {"added": added, "memories": n, "seconds": round(dt)}

    out = PROJECT_ROOT / "results" / f"reingest_{args.prefix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
