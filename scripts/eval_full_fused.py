#!/usr/bin/env python
"""全量融合检索评测：LoCoMo 1986 题，graph_mode=fused（S2 图 = S1 + Concept）。

与无图基线（full_locomo_*.jsonl）和 bypass 图（full_locomo_graph_scoped_*.jsonl）
同口径对照：同一批已 ingest 记忆（full:locomo:{conv}）、同一 answer prompt、
同一上下文预算（max_context_memories）。融合模式 = 节点中心聚合召回
（Entity/Event/Preference/Concept 一次带出节点下全部记忆）与向量结果
同池统一 rerank，不占固定槽位。

用法：
    .venv/bin/python scripts/eval_full_fused.py --eval-only
    .venv/bin/python scripts/eval_full_fused.py --eval-only --conv conv-26   # 冒烟
    .venv/bin/python scripts/eval_full_fused.py --resume results/full_locomo_fused_xxx.jsonl
    .venv/bin/python scripts/eval_full_fused.py --eval-only --config config/locomo_zh.yaml \
        --prefix zhfull --graphs s2,v8cluster   # 中文轨道（与 eval_full.py 的 --config/--prefix 同法）
    .venv/bin/python scripts/eval_full_fused.py --eval-only --config config/locomo_zh.yaml \
        --prefix zhfull --graphs s2,v8cluster --jev-stop \
        --decision-base https://<host>:8443    # Jev 停止准则臂：decider-2b→jevdec / laya-rl-agent→jevlaya

中文轨道注意：
- --config 换配置（数据集 locomo10_zh / mem0 向量库 *_zh / 结果目录 results_zh）
- --prefix zhfull → user_id = zhfull:locomo:{conv}，输出 results_zh/zhfull_fused_locomo_<stamp>.jsonl
  （模板 {prefix}_fused_locomo_<stamp>，与无图基线 {prefix}_locomo_*.jsonl 互斥可区分：
  基线 glob zhfull_locomo_*.jsonl 不会误匹配融合臂，配对定位 A/B 各自唯一模式）
- 图变体与英文终版 B 臂一致取 s2+v8cluster（--graphs s2,v8cluster）；
  answer 预设保持 official（默认，与中文无图基线同口径，只对照图这一变量）
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
from schema_rsi.graph import HugeGraphStore  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402

_progress_lock = threading.Lock()
_done_count = 0

# Jev 停止准则臂的驱动服务身份：以 /v1/decisions 响应的 model 字段为准（不信任域名）。
# model → 输出文件名段：decider-2b → jevdec，laya-rl-agent → jevlaya。
_JEV_MODELS = {"decider-2b": "jevdec", "laya-rl-agent": "jevlaya"}


def _decision_selfcheck(base_url: str) -> str:
    """启动自检：对 decision-base 发一次真实 POST（体例同 LayaClient.decide），取响应 model 字段。

    请求失败 / 非 200 / 无 model 字段 / model 不在允许集合 → 打印真实原因并退出（不绕过）。
    """
    import requests
    url = base_url.rstrip("/") + "/v1/decisions"
    body = {
        "state": {"body": "身份自检（identity self-check）：Question: ping. Context: ping."},
        "questions": {"sufficient": {
            "type": "noul",
            "instructions": "Identity self-check: the context contains the facts needed.",
        }},
    }
    try:
        r = requests.post(url, json=body, timeout=15)
    except requests.RequestException as e:
        print(f"[decision-base] ✗ 自检请求失败：{url}\n    {type(e).__name__}: {e}")
        sys.exit(2)
    model = None
    if r.status_code == 200:
        try:
            model = r.json().get("model")
        except ValueError:
            model = None
    if model not in _JEV_MODELS:
        print(f"[decision-base] ✗ 身份校验失败：model={model!r}（HTTP {r.status_code}），"
              f"允许集合 {{{', '.join(sorted(_JEV_MODELS))}}}，退出。body[:120]={r.text[:120]!r}")
        sys.exit(2)
    print(f"[decision-base] {url} → model={model} ✓（Jev 臂文件名段：{_JEV_MODELS[model]}）")
    return model


def main() -> int:
    eval_only = "--eval-only" in sys.argv
    conv_filter = {
        sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--conv" and i + 1 < len(sys.argv)
    }
    # --config config/locomo_zh.yaml：换配置轨道（数据集/向量库/结果目录隔离）
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else None
    # --prefix zhfull：user_id 与结果文件名前缀（默认 full，与历史英文轨道一致）
    prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "full"
    settings = get_settings(config_path)
    cfg = settings.raw.get("full") or {}
    answer_samples = int(cfg.get("answer_samples", 1))
    workers = int(cfg.get("workers", 8))

    # 获胜者参数（探索轮次的胜出配置）：--graphs s2,v8cluster --per-node 30 --preset official_precise
    def _opt(name, default=None):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    graphs = _opt("--graphs", "s2")
    per_node = int(_opt("--per-node", 15))
    pool = int(_opt("--pool", 30))
    preset = _opt("--preset", "official")
    use_atomic = "--no-atomic" not in sys.argv
    jev_stop = "--jev-stop" in sys.argv
    laya_filter = "--laya-filter" in sys.argv
    tag = _opt("--tag")  # 可选轮次标记；默认不带（文件名模板已含 fused 段）
    # --decision-base <url>：注入 LayaClient(base_url=...) 驱动 jev_stop 判定；
    # 启动即自检，model 字段写入 stdout 与输出 JSON 的每条 metadata（decision_model）
    decision_base = _opt("--decision-base")
    decision_model = _decision_selfcheck(decision_base) if decision_base else None

    # 融合模式覆盖（默认 yaml 仍是 bypass，不改变既有脚本行为）
    ev = settings.raw.setdefault("evaluation", {})
    ev["graph_mode"] = "fused"
    ev["fused_per_node"] = per_node
    ev["fused_pool"] = pool

    print("=" * 72)
    print(f"全量融合评测（graphs={graphs}, per_node={per_node}, pool={pool}, preset={preset}, "
          f"samples={answer_samples}, workers={workers}）")
    print("=" * 72)

    store = HugeGraphStore(settings)
    counts = {lab: store.count_vertices(lab)
              for lab in ("Memory", "Entity", "Event", "Preference", "Concept")}
    # 注意：共用 hugegraph 实例（user 域隔离）时这是全标签计数（含其他轨道顶点）；
    # 按 user_id 前缀的精确计数见 scripts/build_graph_zh.py 的输出
    print(f"图状态（S2，全标签计数，graph={settings.hugegraph.graph}）: {counts}")
    if not counts.get("Concept"):
        print("!! 图里没有 Concept 顶点：先跑 scripts/build_concepts.py")
        return 1

    backend = Mem0Backend(settings)
    pipeline = EvaluationPipeline(
        backend, settings, graph_store=store,
        judge_client=make_judge_client(settings),
    )
    from schema_rsi.evaluation.answerer import Answerer
    if pipeline.answerer is None:
        pipeline.answerer = Answerer(settings)
    pipeline.answerer.preset = preset
    pipeline.retriever.use_atomic = use_atomic
    pipeline.jev_stop = jev_stop
    pipeline.laya_filter = laya_filter
    print(f"[axes] atomic={use_atomic} jev_stop={jev_stop} laya_filter={laya_filter}")
    if decision_base:
        # 注入指定决策服务（_laya() 对 _laya_client 有缓存，先到先得，见 pipeline._laya）
        from schema_rsi.llm.laya import LayaClient
        pipeline._laya_client = LayaClient(settings, base_url=decision_base)
        print(f"[decision] base={decision_base} model={decision_model}")
    if graphs and graphs != "none":
        from schema_rsi.graph.variants import hops_for
        pipeline.fused_hops = hops_for([v.strip() for v in graphs.split(",") if v.strip()])

    ds = LocomoDataset(settings.locomo_path)
    ds.load()

    by_conv: dict[str, list] = {}
    for case in ds.cases:
        conv = str(case.metadata.get("conversation_id"))
        if conv_filter and conv not in conv_filter:
            continue
        by_conv.setdefault(conv, []).append(case)

    if not eval_only:
        print(f"\n[Ingest] {len(by_conv)} conversations")
        for conv, cases in by_conv.items():
            uid = f"{prefix}:locomo:{conv}"
            existing = backend.get_all_memories(user_id=uid)
            if existing:
                print(f"    {conv}: 复用 {len(existing)} 条记忆（跳过）")
                continue
            added = pipeline.ingest_case(cases[0], max_turns=None, user_id=uid)
            print(f"    {conv}: -> +{added} 条记忆")

    resume_path = sys.argv[sys.argv.index("--resume") + 1] if "--resume" in sys.argv else None
    done_ids: set[str] = set()
    if resume_path:
        done_ids = {json.loads(line)["case_id"] for line in open(resume_path, encoding="utf-8")}
        print(f"\n[Resume] {resume_path} 已有 {len(done_ids)} 题结果，跳过")
        out_path = Path(resume_path)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 模板 {prefix}_fused_locomo_<stamp>：fused 段在 locomo 之前，
        # 不落入基线 glob {prefix}_locomo_*.jsonl（否则配对定位会把融合臂误当基线）
        # Jev 臂再按驱动服务插入一段：{prefix}_fused_jevdec|jevlaya_locomo_<stamp>.jsonl
        arm = _JEV_MODELS[decision_model] if (jev_stop and decision_model) else None
        stem = f"{prefix}_fused_{arm}_locomo" if arm else f"{prefix}_fused_locomo"
        name = f"{stem}_{tag}_{stamp}.jsonl" if tag else f"{stem}_{stamp}.jsonl"
        out_path = settings.results_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for conv, cases in by_conv.items():
        uid = f"{prefix}:locomo:{conv}"
        for case in cases:
            if case.case_id not in done_ids:
                tasks.append((case, uid))
    print(f"\n[Evaluate] 待评测 {len(tasks)} QA → {out_path.name}")

    from schema_rsi.llm.rate import AdaptiveLimiter, is_rate_limit_error

    limiter = AdaptiveLimiter(start=8, ceiling=20)

    def run_case(case, uid):
        limiter.acquire()
        try:
            for attempt in range(4):
                try:
                    r = pipeline.evaluate_case(
                        case, graph_enabled=(graphs != "none"), user_id=uid, answer_samples=answer_samples
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
    fused_in_ctx: list[int] = []
    with open(out_path, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(run_case, case, uid): case for case, uid in tasks}
        for fut in as_completed(futures):
            case = futures[fut]
            try:
                r = fut.result()
                if r is not None:
                    d = r.to_dict()
                    if decision_model:
                        # 决策服务身份随每条结果落盘（meta），供下游按驱动服务分组
                        d.setdefault("metadata", {})["decision_model"] = decision_model
                    fout.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                    fout.flush()
                    n = (r.metadata or {}).get("fused_graph_in_context")
                    if isinstance(n, int):
                        fused_in_ctx.append(n)
            except Exception as e:
                print(f"    ⚠ {case.case_id} 失败: {str(e)[:100]}")
            with _progress_lock:
                _done_count += 1
                if _done_count % 100 == 0:
                    rate = _done_count / (time.time() - t_start)
                    eta = (len(tasks) - _done_count) / max(rate, 1e-6) / 60
                    avg_in = sum(fused_in_ctx) / max(len(fused_in_ctx), 1)
                    print(f"    进度 {_done_count}/{len(tasks)} ({rate:.1f} 题/s, "
                          f"并发 {limiter.limit}, 图入选均值 {avg_in:.1f}, ETA {eta:.0f}min)")

    rows = [json.loads(line) for line in open(out_path, encoding="utf-8")]

    def _acc(rs):
        js = [r for r in rs if "judge_correct" in r["metrics"]]
        return (sum(1 for r in js if r["metrics"]["judge_correct"]) / len(js)) if js else float("nan")

    scored = [r for r in rows if r["metadata"].get("category") != "adversarial"]
    adversarial = [r for r in rows if r["metadata"].get("category") == "adversarial"]
    print(f"\n[汇总] 有效 {len(rows)}/{len(tasks)} 题，耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    print(f"    官方口径 J-score（类别1-4，{len(scored)} 题）: {_acc(scored):.1%}")
    if adversarial:
        print(f"    adversarial: {_acc(adversarial):.1%}")
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(str(r["metadata"].get("category")), []).append(r)
    for cat in sorted(by_cat):
        print(f"      {cat:12} {len(by_cat[cat]):5} 题  judge {_acc(by_cat[cat]):.1%}")
    print(f"\n✓ 融合评测完成：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
