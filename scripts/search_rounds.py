#!/usr/bin/env python
"""100 轮多图组合探索跑批器。

评测方法学（最优先设计）：
- dev/test 按 conversation 切分（dev=7 对话 / test=3 对话）：轮次选择只看 dev，
  防止 100 轮在评测集上过拟合；test 留给最终获胜者确认
- 每轮固定题目集 dev_ids（fused 臂全部非 exact 题 + 40% exact 题作回归守卫）
- 内环判分 = glm-5.3-flash 严格精准判分（高并发、按模型独立缓存）；
  最终获胜者用 glm-5.3 非 flash @≤2 并发终审
- r000 校准轮 = 复跑与参考臂完全相同的配置，量化单采样噪声；
  轮次间差异小于噪声带视为无差异

用法：
  .venv/bin/python scripts/search_rounds.py --make-dev          # 生成 dev 集与参考评分
  .venv/bin/python scripts/search_rounds.py                     # 按 config/search_rounds.yaml 跑
  .venv/bin/python scripts/search_rounds.py --max-rounds 5      # 限制本轮数量
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation import EvaluationPipeline  # noqa: E402
from schema_rsi.evaluation.judge import make_judge_client  # noqa: E402
from schema_rsi.evaluation.precision_judge import PrecisionJudge  # noqa: E402
from schema_rsi.graph import HugeGraphStore  # noqa: E402
from schema_rsi.graph.variants import ensure_variant, hops_for  # noqa: E402
from schema_rsi.llm.rate import AdaptiveLimiter, is_rate_limit_error  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402

FUSED_RUN = "results/full_locomo_fused_20260924_055131.jsonl"
FUSED_REGRADE = "results/precision_regrade_fused.jsonl"
SEARCH_DIR = Path("data/search")
OUT = Path("results/search_rounds.jsonl")

_progress_lock = threading.Lock()


def conv_of(case_id: str) -> str:
    return case_id.split("_qa")[0].replace("locomo_", "")


def make_dev(settings) -> None:
    """dev/test 按 conversation 切分 + dev 题目集 + 参考臂（fused）严格-flash 评分。"""
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    convs = sorted({str(c.metadata.get("conversation_id")) for c in ds.cases})
    dev_convs, test_convs = convs[:7], convs[7:]
    print(f"dev convs: {dev_convs}\ntest convs: {test_convs}")

    fused = {json.loads(l)["case_id"]: json.loads(l) for l in open(FUSED_RUN)}
    regrade = {json.loads(l)["case_id"]: json.loads(l) for l in open(FUSED_REGRADE)}
    dev_rows = [r for cid, r in fused.items()
                if conv_of(cid) in dev_convs and (r.get("metadata") or {}).get("category") != "adversarial"]
    non_exact = [r for r in dev_rows if regrade.get(r["case_id"], {}).get("final") != "exact"]
    exact = [r for r in dev_rows if regrade.get(r["case_id"], {}).get("final") == "exact"]
    rng = random.Random(7)
    guard = rng.sample(exact, int(len(exact) * 0.4))
    dev_ids = sorted(r["case_id"] for r in non_exact + guard)
    print(f"dev 题目: {len(dev_rows)}（全量 dev J 题）→ 选中 {len(dev_ids)}"
          f"（非exact {len(non_exact)} + exact守卫 {len(guard)}）")

    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    (SEARCH_DIR / "split.json").write_text(json.dumps(
        {"dev_convs": dev_convs, "test_convs": test_convs}, ensure_ascii=False, indent=1))
    (SEARCH_DIR / "dev_ids.json").write_text(json.dumps(dev_ids))

    # 参考臂评分：fused 答案用 flash 严格判分（lenient-correct 子集），缓存后零成本
    judge = PrecisionJudge(model="glm-5.3-flash",
                           cache_path="data/search/flash_strict_cache.json")
    limiter = AdaptiveLimiter(start=6, ceiling=12)

    def work(row):
        limiter.acquire()
        try:
            for _ in range(4):
                try:
                    out = judge.grade((row.get("metadata") or {}).get("category"), row["question"],
                                      row["expected_answer"] or "", (row.get("predicted_answer") or "")[:400])
                    limiter.maybe_increase()
                    return row["case_id"], out["grade"]
                except Exception as e:  # noqa: BLE001
                    if is_rate_limit_error(e):
                        limiter.on_rate_limited(); time.sleep(4)
                    else:
                        time.sleep(2)
            return row["case_id"], "unknown"
        finally:
            limiter.release()

    ref = {}
    todo = [r for r in dev_rows if r["case_id"] in set(dev_ids) and (r.get("metrics") or {}).get("judge_correct")]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            cid, grade = fut.result()
            ref[cid] = grade
            if i % 100 == 0:
                print(f"  参考评分 {i}/{len(todo)} (conc={limiter.limit})", flush=True)
    judge.flush()
    (SEARCH_DIR / "ref_fused_dev.json").write_text(json.dumps(ref, ensure_ascii=False))
    ex = sum(1 for v in ref.values() if v == "exact")
    print(f"参考臂（fused）dev 严格-flash：exact {ex}/{len(dev_ids)} "
          f"({ex/len(dev_ids):.1%})，判分缓存 {len(judge._cache)}")  # noqa: SLF001


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-dev", action="store_true")
    ap.add_argument("--rounds", default="config/search_rounds.yaml")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--ref", default="data/search/ref_fused_dev.json",
                    help="参考臂逐题严格评分 {case_id: grade}；应与轮次同条件测量")
    ap.add_argument("--ref-round", default=None,
                    help="用某个已完成轮次的逐题结果构建参考臂（写入 --ref 指向的文件）")
    args = ap.parse_args()

    settings = get_settings()
    if args.make_dev:
        make_dev(settings)
        return

    rounds = yaml.safe_load(open(args.rounds, encoding="utf-8"))
    if not isinstance(rounds, list) or not rounds:
        print("!! 轮次配置为空"); return 1  # type: ignore
    dev_ids = json.loads((SEARCH_DIR / "dev_ids.json").read_text())

    if args.ref_round:
        # 同条件参考臂：把指定轮次的逐题 final 作为新参考（校准轮即参考）
        src = SEARCH_DIR / "rounds" / f"{args.ref_round}.jsonl"
        rows = [json.loads(l) for l in open(src)]
        ref = {r["case_id"]: r["final"] for r in rows}
        Path(args.ref).write_text(json.dumps(ref, ensure_ascii=False))
        print(f"[ref] 参考臂 ← 轮次 {args.ref_round}（n={len(ref)}, "
              f"exact={sum(1 for v in ref.values() if v=='exact')}）")
        return
    ref = json.loads(Path(args.ref).read_text())

    # ---- 数据/记忆/图变体（幂等，全部轮共享） ----
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    case_map = {c.case_id: c for c in ds.cases}
    uid_of = {c.case_id: f"full:locomo:{c.metadata.get('conversation_id')}" for c in ds.cases}
    import os

    if os.environ.get("SCHEMA_RSI_VECTOR") == "bm":
        from schema_rsi.memory.chroma_direct import ChromaDirectBackend
        backend = ChromaDirectBackend(settings)
        print("[vector] BigModel 直连库 data/chroma_bm/")
    else:
        backend = Mem0Backend(settings)
    all_memories = []
    for conv in sorted({str(c.metadata.get("conversation_id")) for c in ds.cases}):
        all_memories.extend(backend.get_all_memories(user_id=f"full:locomo:{conv}"))
    print(f"记忆 {len(all_memories)} 条")

    store = HugeGraphStore(settings)
    needed = sorted({v for r in rounds for v in (r.get("graphs") or [])})
    for vid in needed:
        print(f"[variant] {vid}: {ensure_variant(store, vid, all_memories, settings)}")

    # ---- pipeline（fused 模式；轮内动态改 hops/preset/参数） ----
    settings.raw.setdefault("evaluation", {})["graph_mode"] = "fused"
    pipeline = EvaluationPipeline(backend, settings, graph_store=store,
                                  judge_client=make_judge_client(settings))
    from schema_rsi.evaluation.answerer import Answerer
    if pipeline.answerer is None:
        pipeline.answerer = Answerer(settings)  # 轮内要直接改 preset，必须先实体化
    flash_judge = PrecisionJudge(model="glm-5.3-flash",
                                 cache_path="data/search/flash_strict_cache.json")
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in open(OUT)}
        print(f"[Resume] 已完成 {len(done)} 轮")
    todo_rounds = [r for r in rounds if r["id"] not in done][: args.max_rounds]
    print(f"本轮执行 {len(todo_rounds)} 轮，dev 题集 {len(dev_ids)}")

    ev_limiter = AdaptiveLimiter(start=6, ceiling=14)
    j_limiter = AdaptiveLimiter(start=6, ceiling=10)

    def eval_one(case, uid, use_graph, samples):
        ev_limiter.acquire()
        try:
            for attempt in range(5):
                try:
                    r = pipeline.evaluate_case(case, graph_enabled=use_graph, user_id=uid,
                                               answer_samples=samples)
                    ev_limiter.maybe_increase()
                    return r
                except Exception as e:  # noqa: BLE001
                    if is_rate_limit_error(e):
                        ev_limiter.on_rate_limited(); time.sleep(3 * (attempt + 1)); continue
                    raise
            return None
        finally:
            ev_limiter.release()

    def strict_flash(row):
        j_limiter.acquire()
        try:
            for _ in range(4):
                try:
                    out = flash_judge.grade((row.metadata or {}).get("category"), row.question,
                                            row.expected_answer or "", (row.predicted_answer or "")[:400])
                    j_limiter.maybe_increase()
                    return out["grade"]
                except Exception as e:  # noqa: BLE001
                    if is_rate_limit_error(e):
                        j_limiter.on_rate_limited(); time.sleep(4)
                    else:
                        time.sleep(2)
            return "unknown"
        finally:
            j_limiter.release()

    for rnd in todo_rounds:
        rid = rnd["id"]
        graphs = rnd.get("graphs") or []
        ret = rnd.get("retrieval") or {}
        preset = rnd.get("answer_preset", "official")
        use_graph = bool(graphs)
        use_rerank = bool(rnd.get("rerank", True))
        t0 = time.time()
        # 轮内配置注入（rerank / laya 都是轮次轴）
        settings.raw["rerank_enabled"] = use_rerank
        if use_rerank and pipeline.retriever._reranker is None:  # noqa: SLF001
            from schema_rsi.llm.rerank import RerankClient
            pipeline.retriever._reranker = RerankClient(  # noqa: SLF001
                base_url=settings.rerank.base_url, api_key=settings.rerank.api_key,
                model=settings.rerank.model)
        elif not use_rerank:
            pipeline.retriever._reranker = None  # noqa: SLF001
        pipeline.laya_route = bool(rnd.get("laya_route", False))
        pipeline.laya_filter = bool(rnd.get("laya_filter", False))
        pipeline.jev_stop = bool(rnd.get("jev_stop", False))
        pipeline.retriever.use_atomic = bool(rnd.get("atomic", True))  # 提取层重做轴
        pipeline.fused_hops = hops_for(graphs) if graphs else None
        pipeline.answerer.preset = preset
        ec = settings.raw["evaluation"]
        ec["graph_seed_k"] = int(ret.get("seed_k", 5))
        ec["fused_per_node"] = int(ret.get("per_node", 15))
        ec["fused_pool"] = int(ret.get("pool", 30))
        print(f"\n===== round {rid}: graphs={graphs or '无图'} retrieval={ret} preset={preset}")

        results = {}
        samples = int(rnd.get("answer_samples", 1))
        with ThreadPoolExecutor(max_workers=14) as pool:
            futs = {pool.submit(eval_one, case_map[cid], uid_of[cid], use_graph, samples): cid
                    for cid in dev_ids if cid in case_map}
            n_done = 0
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    r = fut.result()
                    if r is not None:
                        results[cid] = r
                except Exception as e:  # noqa: BLE001
                    print(f"    ⚠ {cid} 失败: {str(e)[:90]}")
                n_done += 1
                if n_done % 150 == 0:
                    print(f"    评测 {n_done}/{len(futs)} ({time.time()-t0:.0f}s)", flush=True)

        # 严格-flash 判分（lenient-correct 子集）
        grades = {}
        todo = [(cid, r) for cid, r in results.items() if (r.metrics or {}).get("judge_correct")]
        with ThreadPoolExecutor(max_workers=10) as pool:
            futs = {pool.submit(strict_flash, r): cid for cid, r in todo}
            for fut in as_completed(futs):
                grades[futs[fut]] = fut.result()
        flash_judge.flush()

        final = {}
        for cid, r in results.items():
            if not (r.metrics or {}).get("judge_correct"):
                final[cid] = "wrong"
            else:
                final[cid] = grades.get(cid, "unknown")
        c = Counter(final.values())
        ref_ex = sum(1 for cid in final if cid in ref and ref[cid] == "exact")
        my_ex = sum(1 for cid in final if cid in ref and final[cid] == "exact")
        both = [cid for cid in final if cid in ref]
        better = [cid for cid in both if final[cid] == "exact" and ref[cid] != "exact"]
        worse = [cid for cid in both if final[cid] != "exact" and ref[cid] == "exact"]
        per_cat = {}
        for cid in both:
            cat = (results[cid].metadata or {}).get("category")
            per_cat.setdefault(cat, Counter())[final[cid]] += 1
        rec = {
            "id": rid, "graphs": graphs, "retrieval": ret, "answer_preset": preset,
            "rerank": use_rerank,
            "laya_route": bool(rnd.get("laya_route", False)),
            "laya_filter": bool(rnd.get("laya_filter", False)),
            "jev_stop": bool(rnd.get("jev_stop", False)),
            "atomic": bool(rnd.get("atomic", True)),
            "n": len(both), "exact": my_ex, "ref_exact": ref_ex,
            "delta_exact": my_ex - ref_ex,
            "flips_up": len(better), "flips_down": len(worse),
            "grades": dict(c), "per_category": {k: dict(v) for k, v in per_cat.items()},
            "flip_up_ids": better[:40], "flip_down_ids": worse[:40],
            "graph_in_ctx_mean": (sum((r.metadata or {}).get("fused_graph_in_context") or 0
                                      for r in results.values()) / max(len(results), 1)),
            "duration_s": round(time.time() - t0, 1),
        }
        with _progress_lock:
            with OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # 逐题落盘（参考臂构建 + 事后归因都要用）
        (SEARCH_DIR / "rounds").mkdir(parents=True, exist_ok=True)
        with (SEARCH_DIR / "rounds" / f"{rid}.jsonl").open("w", encoding="utf-8") as fh:
            for cid in sorted(results):
                r = results[cid]
                # via 归因：入选上下文的图候选走了哪些节点（SE-GoS 反馈边权的数据源）
                in_ctx = set((r.metadata or {}).get("graph_context_ids") or [])
                vias = sorted({v for g in (r.graph_memories or []) if g.get("id") in in_ctx
                               for v in (g.get("via") or [])[:2]})[:10]
                fh.write(json.dumps({
                    "case_id": cid,
                    "category": (r.metadata or {}).get("category"),
                    "lenient": bool((r.metrics or {}).get("judge_correct")),
                    "final": final[cid],
                    "graph_in_ctx": (r.metadata or {}).get("fused_graph_in_context"),
                    "graph_vias": vias,
                    "pred": (r.predicted_answer or "")[:250],
                }, ensure_ascii=False) + "\n")
        print(f"  → exact {my_ex}/{len(both)} (参考 {ref_ex}) Δ={my_ex-ref_ex:+d} "
              f"↑{len(better)} ↓{len(worse)} | {dict(c)} | {rec['duration_s']}s")

    print(f"\n✓ 完成，结果追加在 {OUT}")


if __name__ == "__main__":
    main()
