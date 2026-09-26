#!/usr/bin/env python
"""图层价值验证 A/B：Schema S0 vs 无图基线（LoCoMo 全量错题子集）。

流程：
  1. 读取无图基线 JSONL，取 J-score 内（非 adversarial）judge 判错的题
  2. EntityExtractor 从全部 LoCoMo 记忆批量抽实体（缓存 data/graph/entities_cache.json）
  3. GraphBuilder.build_s0 建图（Memory + Entity + MENTIONS）
  4. 对错题子集以 graph_enabled=True 重评（同官方口径），统计翻正率
  5. （可选 --control N）抽 N 个基线答对的题做对照组，量化回归风险

用法：
    .venv/bin/python -u scripts/eval_graph_ab.py results/full_locomo_xxx.jsonl [--control 30]
"""
from __future__ import annotations

import json
import random
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
from schema_rsi.graph import GraphBuilder, HugeGraphStore, make_schema_s0  # noqa: E402
from schema_rsi.graph.extractor import EntityExtractor  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402

_lock = threading.Lock()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    control_n = int(sys.argv[sys.argv.index("--control") + 1]) if "--control" in sys.argv else 0
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 20
    start_conc = int(sys.argv[sys.argv.index("--start-conc") + 1]) if "--start-conc" in sys.argv else 8
    if not args:
        print("用法: eval_graph_ab.py <full_locomo_*.jsonl> [--control 30]")
        return 1
    baseline_path = Path(args[0])
    settings = get_settings()
    cfg = settings.raw.get("full") or {}
    answer_samples = int(cfg.get("answer_samples", 1))

    rows = [json.loads(l) for l in open(baseline_path, encoding="utf-8")]
    scored = [r for r in rows if r["metadata"].get("category") != "adversarial"]
    bad = [r for r in scored if r["metrics"].get("judge_correct") is False]
    print(f"基线: {len(rows)} 题 | J-score 内 {len(scored)} 题 | 错题 {len(bad)}")

    # ---- 数据与记忆 ----
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    backend = Mem0Backend(settings)
    conv_user = {str(c.metadata.get("conversation_id")): f"full:locomo:{c.metadata.get('conversation_id')}" for c in ds.cases}
    case_map = {c.case_id: c for c in ds.cases}
    uid_by_case = {c.case_id: conv_user[str(c.metadata.get("conversation_id"))] for c in ds.cases}

    all_memories = []
    for conv, uid in conv_user.items():
        all_memories.extend(backend.get_all_memories(user_id=uid))
    print(f"图输入记忆: {len(all_memories)} 条（{len(conv_user)} 对话）")

    # ---- 实体抽取（缓存）+ 建图 ----
    store = HugeGraphStore(settings)
    extractor = EntityExtractor(settings)
    t0 = time.time()
    extraction = extractor.extract(all_memories)
    n_ents = sum(len(v) for v in extraction.values())
    print(f"实体抽取: {n_ents} 个实体引用（{time.time() - t0:.0f}s，含缓存命中）")

    builder = GraphBuilder(store)
    report = builder.build_s0(all_memories, extraction, make_schema_s0(), reset=True)
    print(f"建图: vertices={report.vertices_written}（Memory {report.label_counts.get('Memory')}"
          f" + Entity {report.label_counts.get('Entity')}）, edges={report.edges_written},"
          f" {report.duration_s}s")

    # ---- A/B 评测 ----
    pipeline = EvaluationPipeline(
        backend, settings, graph_store=store, judge_client=make_judge_client(settings)
    )
    pipeline.answerer

    control_rows = []
    if control_n:
        correct = [r for r in scored if r["metrics"].get("judge_correct") is True]
        random.seed(42)
        control_rows = random.sample(correct, min(control_n, len(correct)))
        print(f"对照组: {len(control_rows)} 个基线答对的题（量化回归风险）")

    targets = bad + control_rows
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = settings.results_dir / f"graph_ab_{stamp}.jsonl"

    def run_one(row):
        case = case_map.get(row["case_id"])
        if case is None:
            return None
        return pipeline.evaluate_case(
            case, graph_enabled=True, user_id=uid_by_case[case.case_id], answer_samples=answer_samples
        )

    print(f"\n[A/B] graph_enabled=true 重评 {len(targets)} 题（渐进式并发）...")
    from schema_rsi.llm.rate import AdaptiveLimiter, is_rate_limit_error

    limiter = AdaptiveLimiter(start=start_conc, ceiling=workers)
    results = []
    done = 0
    t_start = time.time()
    with open(out_path, "w", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, r): r for r in targets}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                r = fut.result()
                if r is not None:
                    results.append((row, r))
                    fout.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
                    fout.flush()
            except Exception as e:
                print(f"    ⚠ {row['case_id']} 失败: {str(e)[:80]}")
            with _lock:
                done += 1
                if done % 40 == 0:
                    print(f"    进度 {done}/{len(targets)} ({done / (time.time() - t_start):.1f} 题/s)")

    # ---- 汇总 ----
    flipped = [1 for old, new in results
               if old["metrics"].get("judge_correct") is False and new.metrics.get("judge_correct")]
    regressed = [1 for old, new in results
                 if old["metrics"].get("judge_correct") is True and not new.metrics.get("judge_correct")]
    print(f"\n[结果] 耗时 {(time.time() - t_start) / 60:.1f} 分钟，落盘 {out_path.name}")
    print(f"    错题翻正: {len(flipped)}/{len(bad)}（+{len(flipped) / max(len(bad), 1):.1%}）")
    if control_rows:
        print(f"    对照回归: {len(regressed)}/{len(control_rows)}（-{len(regressed) / max(len(control_rows), 1):.1%}）")
    by_cat: dict[str, list] = {}
    for old, new in results:
        if old["metrics"].get("judge_correct") is False and new.metrics.get("judge_correct"):
            by_cat.setdefault(str(new.metadata.get("category")), []).append(new.case_id)
    if by_cat:
        print("    翻正分类别:", {k: len(v) for k, v in by_cat.items()})
        for cat, ids in by_cat.items():
            for i in ids[:3]:
                print(f"      [{cat}] {i}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
