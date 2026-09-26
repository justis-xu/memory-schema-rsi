#!/usr/bin/env python
"""Badcase 归因：对评测 JSONL 里 judge 判 WRONG 的题，追溯丢在哪一环。

归因链（按顺序检查）：
  1. filter_skipped   —— ingest 时该 case 的 session 被内容过滤拦截（pipeline.ingest_errors）
  2. extraction_miss  —— gold 关键词不在该 user 的全部记忆里（提取丢失）
  3. retrieval_miss   —— 在记忆库里、但不在检索 top-k 里（检索/重排问题）
  4. answer_error     —— 在检索结果里、答案仍错（作答/综合问题）
  5. judge_dispute    —— 预测答案与 gold 高度重合却被判错（判分争议，人工复核候选）

用法：
    .venv/bin/python scripts/analyze_badcases.py results/full_locomo_xxx.jsonl [--sample 5]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def _gold_keywords(answer: str) -> list[str]:
    """从 gold 答案取内容词（去停用词），用于库内存在性检查。"""
    stop = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "was", "are", "were",
            "for", "on", "at", "with", "by", "it", "s"}
    words = [w for w in _norm(answer).split() if w not in stop and len(w) > 2]
    return words or _norm(answer).split()[:3]


def main() -> int:
    args = sys.argv[1:]
    if not args or not args[0].endswith(".jsonl"):
        print("用法: analyze_badcases.py <results/*.jsonl> [--sample 5] [--exclude-adversarial]")
        return 1
    path = Path(args[0])
    sample_n = int(args[args.index("--sample") + 1]) if "--sample" in args else 5
    exclude_adv = "--exclude-adversarial" in args

    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    pool = [r for r in rows if r["metadata"].get("category") != "adversarial"] if exclude_adv else rows
    # adversarial 的 gold 常为空串（不可答设计），会污染关键词归因，默认单独报告
    bad = [r for r in pool if r["metrics"].get("judge_correct") is False]
    print(f"共 {len(rows)} 题" + (f"（J-score 口径 {len(pool)} 题）" if exclude_adv else "")
          + f"，judge 判错 {len(bad)} 题（{len(bad) / len(pool):.1%}）")
    if not bad:
        return 0

    # 建 user_id -> 记忆库 全量索引（一次读齐）
    from schema_rsi.config import get_settings
    from schema_rsi.memory import Mem0Backend

    backend = Mem0Backend(get_settings())
    uids = {r["metadata"].get("user_id") for r in bad if r["metadata"].get("user_id")}
    store: dict[str, list[str]] = {}
    for uid in sorted(uids):
        try:
            store[uid] = [m.content for m in backend.get_all_memories(user_id=uid)]
        except Exception as e:
            store[uid] = []
            print(f"⚠ 读 {uid} 记忆失败: {str(e)[:80]}")

    attributed: list[tuple[str, dict]] = []
    for r in bad:
        uid = r["metadata"].get("user_id", "")
        kws = _gold_keywords(r["expected_answer"])
        all_mems = store.get(uid, [])
        in_store = [k for k in kws if any(k in _norm(m) for m in all_mems)]
        retrieved = [m.get("content", "") for m in r.get("retrieved_memories", [])]
        in_retrieved = [k for k in in_store if any(k in _norm(x) for x in retrieved)]
        predicted = _norm(r.get("predicted_answer", ""))
        overlap = [k for k in kws if k in predicted]

        if not all_mems:
            cause = "extraction_miss"  # 库为空（多为过滤全拦）
        elif not in_store:
            cause = "extraction_miss"
        elif not in_retrieved:
            cause = "retrieval_miss"
        elif overlap and len(overlap) >= max(1, len(kws) // 2):
            cause = "judge_dispute"
        else:
            cause = "answer_error"
        attributed.append((cause, r))

    counts = Counter(c for c, _ in attributed)
    print("\n[归因分布]")
    order = ["extraction_miss", "retrieval_miss", "answer_error", "judge_dispute"]
    for c in order:
        if counts.get(c):
            pct = counts[c] / len(bad)
            print(f"  {c:16} {counts[c]:4} 题 ({pct:.0%})")

    print(f"\n[每类样例]（最多 {sample_n} 题/类）")
    for c in order:
        samples = [r for cause, r in attributed if cause == c][:sample_n]
        if not samples:
            continue
        print(f"\n--- {c} ---")
        for r in samples:
            print(f"  {r['case_id']} [{r['metadata'].get('category')}]")
            print(f"    Q: {r['question'][:80]}")
            print(f"    gold: {r['expected_answer'][:80]}")
            print(f"    pred: {str(r.get('predicted_answer', ''))[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
