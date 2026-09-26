#!/usr/bin/env python
"""无图基线评测：graph_enabled=false，完全不启动 HugeGraph，验证 Mem0-only 的效果。

这是 Phase 1 的第一步：先把"测量功能"校准到有区分度（非零），
后续任何 Schema 版本（S0/S1/S2...）都和这条基线比。

子集构造（保证 gold 事实落在 ingest 窗口内，否则指标无意义）：
- LoCoMo      : evidence dia_id 全部在前 locomo_max_turns 轮内的 QA
                （基线子集默认排除 temporal/adversarial：需日期换算/不可答设计）
- LongMemEval : 标准答案（归一化）子串出现在前 lme_max_turns 轮内的 case

用法：
    .venv/bin/python scripts/eval_baseline.py                 # 选 case + ingest + 评测
    .venv/bin/python scripts/eval_baseline.py --skip-ingest   # 记忆已入库，只重新评测
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.base import answer_in_window  # noqa: E402
from schema_rsi.benchmarks.locomo import LocomoDataset, evidence_in_window  # noqa: E402
from schema_rsi.benchmarks.longmemeval import LongMemEvalDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation import EvaluationPipeline  # noqa: E402
from schema_rsi.evaluation.judge import make_judge_client  # noqa: E402
from schema_rsi.evaluation.result import append_jsonl, summarize  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402


def select_locomo(ds: LocomoDataset, num: int, max_turns: int, categories: set[str], skip: set[str]):
    """按 conversation 顺序选 evidence-in-window 的 QA；同 conversation 共享一次 ingest。

    skip 里的 conversation 被排除（如主题触发供应商内容过滤、提取系统性受损的对话）。
    """
    by_conv: dict[str, list] = {}
    for case in ds.cases:
        conv = str(case.metadata.get("conversation_id"))
        if conv in skip:
            continue
        by_conv.setdefault(conv, []).append(case)

    selected = []
    for conv, cases in by_conv.items():
        for case in cases:
            if len(selected) >= num:
                break
            if case.category not in categories:
                continue
            if evidence_in_window(case, max_turns):
                selected.append(case)
        if len(selected) >= num:
            break
    return selected


def select_lme(ds: LongMemEvalDataset, num: int, max_turns: int):
    return [c for c in ds.cases if answer_in_window(c, max_turns)][:num]


def main() -> int:
    skip_ingest = "--skip-ingest" in sys.argv
    settings = get_settings()
    cfg = settings.baseline
    locomo_n = int(cfg.get("locomo_num_cases", 8))
    locomo_turns = int(cfg.get("locomo_max_turns", 150))
    locomo_cats = set(cfg.get("locomo_categories", ["single_hop", "multi_hop"]))
    locomo_skip = set(cfg.get("locomo_skip_conversations", []))
    lme_n = int(cfg.get("lme_num_cases", 4))
    lme_turns = int(cfg.get("lme_max_turns", 200))

    print("=" * 72)
    print("无图基线评测（graph_enabled=false，不依赖 HugeGraph）")
    print("=" * 72)

    # ---- 选 case ----
    locomo = LocomoDataset(settings.locomo_path)
    locomo.load()
    lme = LongMemEvalDataset(settings.longmemeval_path)
    lme.load()
    locomo_cases = select_locomo(locomo, locomo_n, locomo_turns, locomo_cats, locomo_skip)
    lme_cases = select_lme(lme, lme_n, lme_turns)
    print(f"[选例] LoCoMo {len(locomo_cases)}/{locomo_n}（evidence∈前{locomo_turns}轮, "
          f"类别∈{sorted(locomo_cats)}）")
    for c in locomo_cases:
        print(f"    {c.case_id} [{c.category}] evidence={c.evidence}")
    print(f"[选例] LongMemEval-S {len(lme_cases)}/{lme_n}（答案子串∈前{lme_turns}轮）")
    for c in lme_cases:
        print(f"    {c.case_id} [{c.category}] answer='{c.answer[:40]}'")
    if not locomo_cases and not lme_cases:
        print("✗ 没有选出任何 case，请调整 config/default.yaml 的 baseline 参数")
        return 1

    # ---- ingest（LoCoMo 同 conversation 共享一次；LME 每 case 独立）----
    backend = Mem0Backend(settings)
    use_judge = bool(cfg.get("use_judge", True))
    answer_samples = int(cfg.get("answer_samples", 5))
    pipeline = EvaluationPipeline(
        backend, settings, graph_store=None,  # 无图！
        judge_client=make_judge_client(settings) if use_judge else None,
    )

    def ensure_ingest(case, user_id: str, max_turns: int) -> str:
        existing = backend.get_all_memories(user_id=user_id)
        if existing:
            return f"复用已入库的 {len(existing)} 条记忆"
        try:
            added = pipeline.ingest_case(case, max_turns=max_turns, user_id=user_id)
            return f"ingest +{added}"
        except Exception as e:  # 兜底（pipeline 已按 session 容错，这里防意外）
            return f"ingest 失败（跳过该 case）: {str(e)[:100]}"

    conv_user: dict[str, str] = {}
    for case in locomo_cases:
        conv = str(case.metadata.get("conversation_id"))
        conv_user.setdefault(conv, f"baseline:locomo:{conv}")

    print("\n[Ingest]")
    for conv, uid in conv_user.items():
        rep = next(c for c in locomo_cases if str(c.metadata.get("conversation_id")) == conv)
        print(f"    locomo {conv}: {ensure_ingest(rep, uid, locomo_turns)}  (user={uid})")
    lme_user = {c.case_id: f"baseline:lme:{c.case_id}" for c in lme_cases}
    for case in lme_cases:
        print(f"    lme {case.case_id}: {ensure_ingest(case, lme_user[case.case_id], lme_turns)}")
    if getattr(pipeline, "ingest_errors", None):
        print(f"    ⚠ ingest 容错跳过的 session: {len(pipeline.ingest_errors)} 个")
        for err in pipeline.ingest_errors[:5]:
            print(f"      - {err['case_id']} {err['session_id']}: {err['error'][:80]}")

    # ---- 评测（无图）----
    print("\n[Evaluate] graph_enabled=false"
          + (f"，answer_samples={answer_samples}(多数票)" if answer_samples > 1 else ""))
    results = []
    failed = []
    for case in locomo_cases:
        uid = conv_user[str(case.metadata.get("conversation_id"))]
        try:
            results.append(
                pipeline.evaluate_case(case, graph_enabled=False, user_id=uid, answer_samples=answer_samples)
            )
        except Exception as e:
            failed.append(case.case_id)
            print(f"    ⚠ 评测失败跳过 {case.case_id}: {str(e)[:100]}")
    for case in lme_cases:
        try:
            results.append(
                pipeline.evaluate_case(
                    case, graph_enabled=False, user_id=lme_user[case.case_id], answer_samples=answer_samples
                )
            )
        except Exception as e:
            failed.append(case.case_id)
            print(f"    ⚠ 评测失败跳过 {case.case_id}: {str(e)[:100]}")
    if not results:
        print("✗ 没有成功评测的 case")
        return 1

    has_judge = any("judge_correct" in r.metrics for r in results)

    # LLM-judge 已在 pipeline 内随多采样投票完成（judge_client 注入时）
    if use_judge and not has_judge:
        print("⚠ judge 未生效（JUDGE_*/LLM_* 均不可用）")

    print(f"\n{'case_id':34} {'类别':18} {'F1':6} {'contains':8}"
          + (f" {'judge':6}" if has_judge else "")
          + " n_ret  predicted")
    for r in results:
        judge_col = f"{str(r.metrics['judge_correct']):6}" if "judge_correct" in r.metrics else ""
        votes = r.metrics.get("judge_votes")
        vote_note = f"({sum(votes)}/{len(votes)})" if votes else ""
        print(f"{r.case_id:34} {str(r.metadata.get('category')):18} "
              f"{r.metrics['token_f1']:<6} {str(r.metrics['contains']):8} "
              f"{judge_col}{vote_note} "
              f"{len(r.retrieved_memories):<5} {r.predicted_answer[:42]!r}")

    locomo_res = [r for r in results if r.benchmark == "locomo"]
    lme_res = [r for r in results if r.benchmark == "longmemeval_s"]
    print("\n[汇总]")
    if locomo_res:
        print(f"    LoCoMo      : {summarize(locomo_res)}")
    if lme_res:
        print(f"    LongMemEval : {summarize(lme_res)}")
    print(f"    总体        : {summarize(results)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = settings.results_dir / f"baseline_mem0only_{stamp}.jsonl"
    append_jsonl(results, path)
    print(f"\n✓ 无图基线评测完成（{len(results)} cases" + (f"，{len(failed)} 个失败跳过" if failed else "") + f"），结果已写入 {path}")
    print("  复跑（不重新 ingest）: .venv/bin/python scripts/eval_baseline.py --skip-ingest")
    print("  对照实验（带图）: 先 ./scripts/start_hugegraph.sh，再运行 smoke_e2e.py 或后续 Schema 实验")
    return 0


if __name__ == "__main__":
    sys.exit(main())
