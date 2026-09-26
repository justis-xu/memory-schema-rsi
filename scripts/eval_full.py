#!/usr/bin/env python
"""全量无图基线评测：LoCoMo 全部 10 conversations / 全部 QA（含全部类别）。

设计（长跑友好）：
- ingest 按 conversation 串行、按 user_id（full:locomo:{conv}）持久化 —— 断点可续：
  已有记忆的 conversation 自动跳过；中断后直接重跑即可。
- 评测并行（full.workers 线程），逐题落盘 JSONL —— 中断后已写的结果不丢。
- 分类别汇总（对齐 LoCoMo 官方按 single_hop/multi_hop/temporal/open_domain/adversarial 报告）。

用法：
    .venv/bin/python scripts/eval_full.py              # ingest（缺的）+ 全量评测
    .venv/bin/python scripts/eval_full.py --eval-only  # 只评测（ingest 已完成）
注意：conv-26 与其他 conversation 一样参与全量评测；历史上的内容过滤判断已被完整重跑推翻。
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

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation import EvaluationPipeline  # noqa: E402
from schema_rsi.evaluation.judge import make_judge_client  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402

_progress_lock = threading.Lock()
_done_count = 0


def main() -> int:
    eval_only = "--eval-only" in sys.argv
    # --conv conv-26 [--conv conv-41 ...]：只跑指定 conversation（单对话先验证速度/效果）
    conv_filter = {
        a for i, a in enumerate(sys.argv) if a == "--conv" and i + 1 < len(sys.argv)
        for a in [sys.argv[i + 1]]
    }
    # --config config/locomo_zh.yaml：换配置轨道（数据集/向量库/结果目录隔离）
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else None
    # --prefix zhfull：user_id 与结果文件名前缀（默认 full，与历史英文轨道一致）
    prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "full"
    settings = get_settings(config_path)
    cfg = settings.raw.get("full") or {}
    answer_samples = int(cfg.get("answer_samples", 1))
    workers = int(cfg.get("workers", 8))

    print("=" * 72)
    print(f"全量无图基线（graph_enabled=false, samples={answer_samples}, workers={workers}）")
    print("=" * 72)

    backend = Mem0Backend(settings)
    pipeline = EvaluationPipeline(
        backend, settings, graph_store=None,  # 无图
        judge_client=make_judge_client(settings),
    )
    pipeline.answerer  # 提前初始化，避免评测线程懒加载竞态

    ds = LocomoDataset(settings.locomo_path)
    ds.load()

    by_conv: dict[str, list] = {}
    for case in ds.cases:
        conv = str(case.metadata.get("conversation_id"))
        if conv_filter and conv not in conv_filter:
            continue
        by_conv.setdefault(conv, []).append(case)

    # ---- 阶段 1：ingest（按 conversation，断点可续）----
    if not eval_only:
        print(f"\n[Ingest] {len(by_conv)} conversations / {ds.count()} QA")
        for conv, cases in by_conv.items():
            uid = f"{prefix}:locomo:{conv}"
            existing = backend.get_all_memories(user_id=uid)
            if existing:
                print(f"    {conv}: 复用 {len(existing)} 条记忆（跳过）")
                continue
            t0 = time.time()
            added = pipeline.ingest_case(cases[0], max_turns=None, user_id=uid)  # 全量窗口
            n_turns = cases[0].history_stats()["turns"]
            print(f"    {conv}: {n_turns} turns -> +{added} 条记忆 "
                  f"({time.time() - t0:.0f}s, QA={len(cases)})")

    # ---- 阶段 2：并行评测，逐题落盘（渐进式并发 + 断点续跑）----
    resume_path = sys.argv[sys.argv.index("--resume") + 1] if "--resume" in sys.argv else None
    done_ids: set[str] = set()
    if resume_path:
        done_ids = {json.loads(line)["case_id"] for line in open(resume_path, encoding="utf-8")}
        print(f"\n[Resume] {resume_path} 已有 {len(done_ids)} 题结果，跳过")
        out_path = Path(resume_path)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = settings.results_dir / f"{prefix}_locomo_{stamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for conv, cases in by_conv.items():
        uid = f"{prefix}:locomo:{conv}"
        for case in cases:
            if case.case_id not in done_ids:
                tasks.append((case, uid))
    print(f"\n[Evaluate] 待评测 {len(tasks)} QA（渐进式并发 8→20，限流降档 15/10），"
          f"逐题写入 {out_path.name}")

    from schema_rsi.llm.rate import AdaptiveLimiter, is_rate_limit_error

    limiter = AdaptiveLimiter(start=8, ceiling=20)

    def run_case(case, uid):
        limiter.acquire()
        try:
            for attempt in range(4):
                try:
                    r = pipeline.evaluate_case(
                        case, graph_enabled=False, user_id=uid, answer_samples=answer_samples
                    )
                    limiter.maybe_increase()
                    return r
                except Exception as e:
                    if is_rate_limit_error(e):
                        new_limit = limiter.on_rate_limited()
                        print(f"    ⚡ 限流，并发降至 {new_limit}，重试 {case.case_id}")
                        time.sleep(3 * (attempt + 1))
                        continue
                    raise
            return None
        finally:
            limiter.release()

    t_start = time.time()
    global _done_count
    _done_count = 0
    with open(out_path, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(run_case, case, uid): (case, uid) for case, uid in tasks}
        for fut in as_completed(futures):
            case, uid = futures[fut]
            try:
                r = fut.result()
                if r is not None:
                    fout.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
                    fout.flush()
            except Exception as e:
                print(f"    ⚠ {case.case_id} 失败: {str(e)[:100]}")
            with _progress_lock:
                _done_count += 1
                if _done_count % 100 == 0:
                    rate = _done_count / (time.time() - t_start)
                    eta = (len(tasks) - _done_count) / max(rate, 1e-6) / 60
                    print(f"    进度 {_done_count}/{len(tasks)} ({rate:.1f} 题/s, "
                          f"并发 {limiter.limit}, ETA {eta:.0f}min)")

    # ---- 阶段 3：汇总（总体 + 分类别 + 分 conversation；adversarial 按官方口径不计分，单列）----
    rows = [json.loads(line) for line in open(out_path, encoding="utf-8")]
    judge = [r for r in rows if "judge_correct" in r["metrics"]]
    contains = sum(1 for r in rows if r["metrics"].get("contains"))

    def _acc(rs):
        js = [r for r in rs if "judge_correct" in r["metrics"]]
        return (sum(1 for r in js if r["metrics"]["judge_correct"]) / len(js)) if js else float("nan")

    scored = [r for r in rows if r["metadata"].get("category") != "adversarial"]
    adversarial = [r for r in rows if r["metadata"].get("category") == "adversarial"]
    print(f"\n[汇总] 有效 {len(rows)}/{len(tasks)} 题，耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    if judge:
        print(f"    官方口径 J-score（类别1-4，{len(scored)} 题）: {_acc(scored):.1%}")
        if adversarial:
            print(f"    adversarial（官方不计分，{len(adversarial)} 题）: {_acc(adversarial):.1%}")
        print(f"    全部 {len(rows)} 题 contains: {contains / len(rows):.1%}")

    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(str(r["metadata"].get("category")), []).append(r)
    print("    分类别:")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        js = [r for r in rs if "judge_correct" in r["metrics"]]
        if js:
            acc = sum(1 for r in js if r["metrics"]["judge_correct"]) / len(js)
            print(f"      {cat:12} {len(rs):5} 题  judge {acc:.1%}")
    by_conv_rows: dict[str, list] = {}
    for r in rows:
        conv = r["case_id"].split("_qa")[0].replace("locomo_", "")
        by_conv_rows.setdefault(conv, []).append(r)
    print("    分 conversation:")
    for conv in sorted(by_conv_rows):
        rs = by_conv_rows[conv]
        js = [r for r in rs if "judge_correct" in r["metrics"]]
        if js:
            acc = sum(1 for r in js if r["metrics"]["judge_correct"]) / len(js)
            print(f"      {conv:10} {len(rs):4} 题  judge {acc:.1%}")

    print(f"\n✓ 全量评测完成：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
