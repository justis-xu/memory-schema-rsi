#!/usr/bin/env python
"""中文记忆库语言污染修复驱动：按序重灌 7 个被污染的对话（2026-09-26）。

背景：data/vector_store_zh/（collection schema_rsi_memories_zh）里 616/1993 条英文，
集中在 zhfull conv-41/42/47/48/49/50 与 nf53 conv-42。根因是 mem0 官方提取 prompt 无
中文强制，flash 在部分对话上整段翻转英文。修复 = config/locomo_zh.yaml 的
mem0.fact_extraction_prompt（简体中文硬指令，v2，冒烟已过：conv-47 session_1 全中文）
+ 删除污染 user（已由 reset_memories 清零）+ 本脚本按序重灌。

行为：
- 按下表顺序逐个调 scripts/ingest_conv.py（与 full run 完全一致的 ingest 代码路径），
  逐个打印 +N 条；每灌完一个立即做语言统计（每条记忆须含 CJK 字符），非中文条数>0 打 WARNING。
- 幂等：目标已有记忆时 ingest_conv.py 自身跳过（scripts/ingest_conv.py:66-68），
  中断后重跑安全；--force 先 reset 该 user 再灌（会重烧该对话的全部提取调用，慎用）。
- 任一目标失败（ingest_conv 非零退出）立即停止——避免在异常状态下继续消耗额度，
  修复后直接重跑本脚本即可续灌（幂等）。
- 退出码：0 = 7 个目标全部灌完且全中文；1 = 有失败或检出非中文条。

调用量预算：zhfull 六个对话约 ~400 次 glm-5.3-flash + nf53 conv-42 约 ~60 次 glm-5.3。
（不要顺手把它当全量评测跑——本脚本只做 ingest，不做任何 regrade。）

用法：
  .venv/bin/python scripts/repair_zh_store.py            # 幂等续灌
  .venv/bin/python scripts/repair_zh_store.py --force    # 全部先删再灌（重烧额度）
  .venv/bin/python scripts/repair_zh_store.py --dry-run  # 只打印计划与当前库存
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 关 mem0 PostHog 噪声，保 stdout 干净

CONFIG = "config/locomo_zh.yaml"

# 重灌计划（有序）：(prefix, conv, 模型覆盖)。None = 用 config 的 flash 档（glm-5.3-flash）。
REPAIR_PLAN = [
    ("zhfull", "conv-41", None),        # 污染最重 155 英/210
    ("zhfull", "conv-42", None),        # 轻微
    ("zhfull", "conv-47", None),        # 最重的英文翻转对话 205 英/214
    ("zhfull", "conv-48", None),        # 轻微
    ("zhfull", "conv-49", None),        # 102 英/159
    ("zhfull", "conv-50", None),        # 轻微
    ("nf53", "conv-42", "glm-5.3"),     # 非 flash 对照库 122 英/207，重灌用 glm-5.3
]

_CJK = re.compile(r"[\u4e00-\u9fff]")


def lang_stats(backend, uid: str) -> tuple[int, int, list[str]]:
    """返回 (中文条数, 非中文条数, 非中文样例≤5)。"""
    mems = backend.get_all_memories(user_id=uid)
    bad = [m.content for m in mems if not _CJK.search(m.content)]
    return len(mems) - len(bad), len(bad), bad[:5]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="每个目标先 reset_memories 再灌（重烧该对话全部提取调用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与当前库存，不调 ingest")
    args = ap.parse_args()

    from schema_rsi.config import get_settings
    from schema_rsi.memory import Mem0Backend

    settings = get_settings(CONFIG)
    backend = Mem0Backend(settings)

    print(f"[repair_zh] 计划 {len(REPAIR_PLAN)} 个目标（config={CONFIG}）：")
    for prefix, conv, model in REPAIR_PLAN:
        uid = f"{prefix}:locomo:{conv}"
        zh, en, _ = lang_stats(backend, uid)
        model_note = f" 模型={model}" if model else " 模型=flash(config)"
        print(f"  {uid:26} 当前库存 {zh + en:4} 条（非中文 {en}）{model_note}")

    if args.dry_run:
        print("[repair_zh] --dry-run：不执行。")
        return 0

    if args.force:
        print("[repair_zh] --force：逐个 reset 后重灌。")
        for prefix, conv, _ in REPAIR_PLAN:
            uid = f"{prefix}:locomo:{conv}"
            backend.reset_memories(user_id=uid)
            print(f"  reset {uid} done")

    failures: list[str] = []
    polluted: list[str] = []
    for i, (prefix, conv, model) in enumerate(REPAIR_PLAN, 1):
        uid = f"{prefix}:locomo:{conv}"
        cmd = [sys.executable, "scripts/ingest_conv.py",
               "--config", CONFIG, "--prefix", prefix, "--conv", conv]
        if model:
            cmd += ["--model", model]
        print(f"\n[repair_zh {i}/{len(REPAIR_PLAN)}] {uid} → {' '.join(cmd[2:])}")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)  # 透传 stdout：逐个打印 +N 条
        if proc.returncode != 0:
            print(f"[repair_zh] {uid} ingest_conv 退出码 {proc.returncode}，停止"
                  f"（修复后重跑本脚本，已完成目标会跳过）")
            failures.append(f"{uid}: ingest_conv rc={proc.returncode}")
            break

        zh, en, samples = lang_stats(backend, uid)
        print(f"[repair_zh] {uid}: 库存 {zh + en} 条，中文 {zh}，非中文 {en}")
        for s in samples:
            print(f"    非中文样例: {s[:80]}")
        if en:
            polluted.append(f"{uid}: {en} 条非中文")

    print("\n=== REPAIR SUMMARY (JSON) ===")
    summary = {
        "config": CONFIG,
        "targets": [f"{p}:locomo:{c}" for p, c, _ in REPAIR_PLAN],
        "failures": failures,
        "language_polluted": polluted,
    }
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("=== END SUMMARY ===")
    if failures or polluted:
        print("[repair_zh] 存在失败或非中文条，退出码 1。")
        return 1
    print("[repair_zh] 7 个目标全部重灌完成且全中文。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
