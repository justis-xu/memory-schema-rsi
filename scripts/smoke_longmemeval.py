#!/usr/bin/env python
"""LongMemEval-S 冒烟测试：load → parse → count → 统计 → 打印完整 case。

注意：265MB JSON 加载峰值内存 ~2-2.5GB（进程退出即释放），首次加载约需数十秒。
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.longmemeval import LongMemEvalDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    print(f"数据文件: {settings.longmemeval_path}")
    ds = LongMemEvalDataset(settings.longmemeval_path)

    t0 = time.perf_counter()
    try:
        ds.load()
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return 1
    elapsed = time.perf_counter() - t0
    # macOS 的 ru_maxrss 单位是字节，Linux 是 KB
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024 * 1024 if sys.platform == "darwin" else 1024
    )
    print(f"加载耗时: {elapsed:.1f}s, 峰值内存: {peak_mb:.0f} MB")

    stats = ds.stats()
    print(f"\n统计: {stats}")
    print(f"case 总数: {ds.count()}")
    print(f"sample ids: {[c.case_id for c in ds.sample(5)]}")

    print("\n打印第一条完整 case（长历史按 session 截断展示）:")
    ds.print_case()

    case = ds.cases[0]
    assert case.metadata.get("raw"), "原始 record 必须保留在 metadata['raw']"
    assert case.history, "history 应非空"
    print(f"\n✓ LongMemEval-S loaded ({stats['questions']} questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
