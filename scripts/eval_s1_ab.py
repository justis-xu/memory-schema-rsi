#!/usr/bin/env python
"""Schema S0 vs S1 对比：同一批错题上对比两种 Schema 的翻正率。

前置条件：
- 无图基线 JSONL（full_locomo_*.jsonl）
- S0 结果 JSONL（graph_ab_*.jsonl，或从 S0 全量结果提取错题）

流程：
  1. StructuredExtractor 抽取实体+事件+偏好+关系（缓存 data/graph/structured_cache.json）
  2. GraphBuilder.build_s1 建图（覆盖 S0 图）
  3. 对基线错题以 graph_enabled=True 重评（同 S0 实验的题目集合）
  4. 输出 S0 vs S1 翻正对比表

用法：
    .venv/bin/python -u scripts/eval_s1_ab.py <baseline.jsonl> <s0_result.jsonl> [--workers 3 --start-conc 2]
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
from schema_rsi.graph import GraphBuilder, HugeGraphStore, make_schema_s1  # noqa: E402
from schema_rsi.graph.extractor_s1 import StructuredExtractor  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402

_lock = threading.Lock()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    start_conc = int(sys.argv[sys.argv.index("--start-conc") + 1]) if "--start-conc" in sys.argv else 2
    if len(args) < 2:
        print("用法: eval_s1_ab.py <baseline.jsonl> <s0_result.jsonl>")
        return 1
    baseline_path, s0_path = Path(args[0]), Path(args[1])
    settings = get_settings()

    # 读取基线和 S0 结果
    baseline = {r["case_id"]: r for r in (json.loads(l) for l in open(baseline_path, encoding="utf-8"))}
    s0_results = {r["case_id"]: r for r in (json.loads(l) for l in open(s0_path, encoding="utf-8"))}

    # 目标题集合：基线错题（J-score 内）
    bad_ids = {cid for cid, r in baseline.items()
               if r["metadata"].get("category") != "adversarial"
               and r["metrics"].get("judge_correct") is False}
    print(f"基线错题: {len(bad_ids)} | S0 已评测: {len(s0_results)}")

    # ---- 数据与记忆 ----
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    backend = Mem0Backend(settings)
    conv_user = {str(c.metadata.get("conversation_id")):
                 f"full:locomo:{c.metadata.get('conversation_id')}" for c in ds.cases}
    case_map = {c.case_id: c for c in ds.cases}
    uid_by_case = {c.case_id: conv_user[str(c.metadata.get("conversation_id"))] for c in ds.cases}

    all_memories = []
    for uid in conv_user.values():
        all_memories.extend(backend.get_all_memories(user_id=uid))
    print(f"图输入记忆: {len(all_memories)} 条")

    # ---- S1 提取 + 建图 ----
    store = HugeGraphStore(settings)
    extractor = StructuredExtractor(settings)
    t0 = time.time()
    extraction = extractor.extract(all_memories)
    n_e = sum(len(v.get("entities", [])) for v in extraction.values())
    n_v = sum(len(v.get("events", [])) for v in extraction.values())
    n_p = sum(len(v.get("preferences", [])) for v in extraction.values())
    n_r = sum(len(v.get("relationships", [])) for v in extraction.values())
    print(f"S1 提取: {n_e} 实体 / {n_v} 事件 / {n_p} 偏好 / {n_r} 关系 ({time.time()-t0:.0f}s)")

    builder = GraphBuilder(store)
    report = builder.build_s1(all_memories, extraction, make_schema_s1(), reset=True)
    print(f"S1 建图: {report.label_counts} | edges={report.edges_written} | {report.duration_s}s")

    # ---- 评测（graph on, S1 图）----
    pipeline = EvaluationPipeline(
        backend, settings, graph_store=store, judge_client=make_judge_client(settings)
    )
    pipeline.answerer

    targets = [baseline[cid] for cid in bad_ids if cid in case_map]
    print(f"\n[S1 评测] {len(targets)} 错题，graph_enabled=True（S1 图）...")

    from schema_rsi.llm.rate import AdaptiveLimiter, is_rate_limit_error

    limiter = AdaptiveLimiter(start=start_conc, ceiling=workers, ladder=(3, 2), up_interval=60)

    def run_one(row):
        case = case_map.get(row["case_id"])
        if case is None:
            return None
        limiter.acquire()
        try:
            for attempt in range(4):
                try:
                    r = pipeline.evaluate_case(
                        case, graph_enabled=True, user_id=uid_by_case[case.case_id], answer_samples=1
                    )
                    limiter.maybe_increase()
                    return r
                except Exception as e:
                    if is_rate_limit_error(e):
                        limiter.on_rate_limited()
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise
            return None
        finally:
            limiter.release()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = settings.results_dir / f"s1_ab_{stamp}.jsonl"
    s1_results: dict[str, dict] = {}
    done = 0
    t_start = time.time()
    with open(out_path, "w", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, r): r for r in targets}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                r = fut.result()
                if r:
                    s1_results[r.case_id] = r.to_dict()
                    fout.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
                    fout.flush()
            except Exception as e:
                print(f"    ⚠ {row['case_id']}: {str(e)[:80]}")
            with _lock:
                done += 1
                if done % 20 == 0:
                    print(f"    进度 {done}/{len(targets)} ({done/(time.time()-t_start):.1f} 题/s)")

    # ---- S0 vs S1 对比 ----
    print(f"\n{'='*60}")
    print(f"Schema S0 vs S1 翻正对比（{len(bad_ids)} 基线错题）")
    print(f"{'='*60}")
    s0_flipped = set()
    s1_flipped = set()
    for cid in bad_ids:
        s0_r = s0_results.get(cid, {})
        s1_r = s1_results.get(cid, {})
        if s0_r.get("metrics", {}).get("judge_correct"):
            s0_flipped.add(cid)
        if s1_r.get("metrics", {}).get("judge_correct"):
            s1_flipped.add(cid)

    both = s0_flipped & s1_flipped
    only_s0 = s0_flipped - s1_flipped
    only_s1 = s1_flipped - s0_flipped
    neither = bad_ids - s0_flipped - s1_flipped

    print(f"  S0 翻正: {len(s0_flipped)} ({len(s0_flipped)/len(bad_ids):.1%})")
    print(f"  S1 翻正: {len(s1_flipped)} ({len(s1_flipped)/len(bad_ids):.1%})")
    print(f"  两者都翻正: {len(both)}")
    print(f"  仅 S0: {len(only_s0)}")
    print(f"  仅 S1: {len(only_s1)} ← S1 的增量贡献")
    print(f"  都没翻正: {len(neither)}")

    if only_s1:
        print(f"\n  S1 独有翻正题:")
        for cid in sorted(only_s1)[:8]:
            r = baseline[cid]
            print(f"    {cid} [{r['metadata'].get('category')}] {r['question'][:60]}")

    print(f"\n落盘: {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
