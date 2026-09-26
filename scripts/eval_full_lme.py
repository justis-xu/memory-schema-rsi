#!/usr/bin/env python
"""LongMemEval-S 全量无图基线（mem0 官方 benchmark 口径）。

- ingest：每 case 的完整 haystack 按 session 灌入（带日期锚），user_id=full:lme:{qid}，
  断点可续（已有记忆的 case 跳过）。每 case 约 5-7 分钟（官方提取，无思考）。
- 评测：并行（full.workers），官方 LME answer/judge prompt（question_date 锚定、
  yes/no 判卷），逐题落盘 JSONL。
- 汇总：总体 + 分 question_type。

用法：
    .venv/bin/python -u scripts/eval_full_lme.py --num 20    # 前 20 个 case
    .venv/bin/python -u scripts/eval_full_lme.py --num 500   # 全量（S 版共 500 题）
    .venv/bin/python -u scripts/eval_full_lme.py --eval-only --num 20
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.longmemeval import LongMemEvalDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation import EvaluationPipeline  # noqa: E402
from schema_rsi.evaluation.judge import make_judge_client  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402

_lock = threading.Lock()
_done = 0


def _reject_obvious_partial(case, existing: list) -> None:
    expected = {str(s["session_id"]) for s in case.history if s.get("turns")}
    observed = {str(r.metadata.get("session_id")) for r in existing}
    missing = expected - observed
    # Old runs lack a session completion manifest; zero-fact sessions are
    # possible, so this catches only large, obvious gaps such as a 429 burst.
    if len(missing) > max(5, int(len(expected) * 0.2)):
        raise RuntimeError(
            f"{case.case_id} has only {len(observed)}/{len(expected)} "
            "sessions represented in existing memory; likely partial ingest. "
            "Do not evaluate or skip this user until its ingest is repaired "
            "or rebuilt in an isolated namespace."
        )


def main() -> int:
    eval_only = "--eval-only" in sys.argv
    num = 500
    if "--num" in sys.argv:
        num = int(sys.argv[sys.argv.index("--num") + 1])
    stratified = "--stratified" in sys.argv  # 按 question_type 轮转分层取样（代表性）
    settings = get_settings()
    cfg = settings.raw.get("full") or {}
    answer_samples = int(cfg.get("answer_samples", 1))
    workers = int(cfg.get("workers", 8))

    print("=" * 72)
    print(f"LongMemEval-S 官方口径（前 {num} case, samples={answer_samples}, workers={workers}）")
    print("=" * 72)

    backend = Mem0Backend(settings)
    pipeline = EvaluationPipeline(
        backend, settings, graph_store=None, judge_client=make_judge_client(settings)
    )
    pipeline.answerer  # 预热，避免线程竞态

    ds = LongMemEvalDataset(settings.longmemeval_path)
    ds.load()
    if stratified:
        # 按 question_type 轮转取样，保证类型代表性（数据集本身按类型分块，顺序取样会偏科）
        by_type: dict[str, list] = {}
        for c in ds.cases:
            by_type.setdefault(str(c.category), []).append(c)
        cases = []
        round_i = 0
        while len(cases) < min(num, len(ds.cases)):
            added = False
            for qt in sorted(by_type):
                if round_i < len(by_type[qt]) and len(cases) < num:
                    cases.append(by_type[qt][round_i])
                    added = True
            if not added:
                break
            round_i += 1
        print(f"[Stratified] 按 question_type 轮转取 {len(cases)} case："
              f"{ {t: sum(1 for c in cases if c.category == t) for t in sorted(set(c.category for c in cases))} }")
    else:
        cases = ds.cases[:num]

    if not eval_only:
        print(f"\n[Ingest] {len(cases)} cases（每 case 完整 haystack，断点可续）")
        for i, case in enumerate(cases):
            uid = f"full:lme:{case.case_id}"
            existing = backend.get_all_memories(user_id=uid)
            if existing:
                _reject_obvious_partial(case, existing)
                continue
            t0 = time.time()
            error_start = len(getattr(pipeline, "ingest_errors", []))
            added = pipeline.ingest_case(case, max_turns=None, user_id=uid)
            new_errors = getattr(pipeline, "ingest_errors", [])[error_start:]
            if new_errors:
                # 内容过滤跳过的 session 是已知且不可避免的（供应商约束），
                # 只要不构成严重不完整（_reject_obvious_partial 已兜底 >20%），继续评测
                filter_skips = sum(1 for e in new_errors if "contentFilter" in str(e.get("error", "")))
                print(f"    [{i + 1}/{len(cases)}] {case.case_id}: "
                      f"⚠ {len(new_errors)} session 跳过（{filter_skips} 个内容过滤），继续")
            turns = case.history_stats()["turns"]
            print(f"    [{i + 1}/{len(cases)}] {case.case_id}: {turns} turns -> +{added} 条 "
                  f"({time.time() - t0:.0f}s)", flush=True)
    else:
        for case in cases:
            existing = backend.get_all_memories(user_id=f"full:lme:{case.case_id}")
            if not existing:
                raise RuntimeError(f"{case.case_id} has no memory for --eval-only")
            _reject_obvious_partial(case, existing)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = settings.results_dir / f"full_lme_{stamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Evaluate] {len(cases)} 题，逐题写入 {out_path.name}")

    t_start = time.time()
    global _done
    with open(out_path, "w", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                pipeline.evaluate_case, c, graph_enabled=False,
                user_id=f"full:lme:{c.case_id}", answer_samples=answer_samples,
            ): c
            for c in cases
        }
        for fut in as_completed(futures):
            case = futures[fut]
            try:
                r = fut.result()
                fout.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
                fout.flush()
            except Exception as e:
                print(f"    ⚠ {case.case_id} 失败: {str(e)[:100]}")
            with _lock:
                _done += 1
                if _done % 20 == 0:
                    rate = _done / (time.time() - t_start)
                    eta = (len(cases) - _done) / max(rate, 1e-6) / 60
                    print(f"    进度 {_done}/{len(cases)} ({rate:.2f} 题/s, ETA {eta:.0f}min)")

    rows = [json.loads(line) for line in open(out_path, encoding="utf-8")]
    judge = [r for r in rows if "judge_correct" in r["metrics"]]
    print(f"\n[汇总] 有效 {len(rows)}/{len(cases)}，评测耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    if judge:
        acc = sum(1 for r in judge if r["metrics"]["judge_correct"]) / len(judge)
        print(f"    总体 judge（yes）准确率: {acc:.1%}")
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(str(r["metadata"].get("category")), []).append(r)
    print("    分 question_type:")
    for qt in sorted(by_type):
        rs = by_type[qt]
        js = [r for r in rs if "judge_correct" in r["metrics"]]
        if js:
            acc = sum(1 for r in js if r["metrics"]["judge_correct"]) / len(js)
            print(f"      {qt:26} {len(rs):4} 题  judge {acc:.1%}")

    print(f"\n✓ 完成：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
