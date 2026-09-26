#!/usr/bin/env python
"""LoCoMo 数据集冒烟测试：load → parse → count → 统计 → 打印完整 case。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    print(f"数据文件: {settings.locomo_path}")
    ds = LocomoDataset(settings.locomo_path)

    t0 = time.perf_counter()
    try:
        ds.load()
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return 1
    print(f"加载耗时: {time.perf_counter() - t0:.2f}s")

    stats = ds.stats()
    print(f"\n统计: {stats}")
    print(f"case 总数: {ds.count()}")
    print(f"sample ids: {[c.case_id for c in ds.sample(3)]}")

    print("\n打印第一条完整 case（长历史按 session 截断展示）:")
    ds.print_case()

    case = ds.cases[0]
    assert case.metadata.get("raw"), "原始 record 必须保留在 metadata['raw']"
    assert case.history and case.history[0]["turns"], "history 应非空"
    print(f"\n✓ LoCoMo loaded ({stats['conversations']} conversations, {stats['qa_total']} QA)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
