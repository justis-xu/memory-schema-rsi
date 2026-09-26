#!/usr/bin/env python
"""最小端到端冒烟：LoCoMo 1 个 conversation（截断）+ LongMemEval-S 3 个 case（截断）
   → Mem0 ingest → GraphBuilder → HugeGraph → 打印 memories / vertices / edges
   → 小规模评测（graph_enabled=false/true 对照，写 results/*.jsonl）。

⚠️ 这是 infrastructure smoke test：GraphBuilder 的 V0 策略（每条 Memory 一个顶点 +
   同 session 相邻 RELATED_TO）只是临时测试关系，不是正式 Graph Schema 设计。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.benchmarks.longmemeval import LongMemEvalDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.evaluation import EvaluationPipeline  # noqa: E402
from schema_rsi.evaluation.result import summarize  # noqa: E402
from schema_rsi.graph import GraphBuilder, HugeGraphStore, make_test_schema  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402

BANNER = "=" * 72 + "\n  INFRASTRUCTURE SMOKE TEST — 临时测试 Schema，非正式 Graph Schema 设计\n" + "=" * 72


def main() -> int:
    print(BANNER)
    settings = get_settings()
    e2e = settings.e2e or {}
    locomo_idx = int(e2e.get("locomo_conversation_index", 0))
    locomo_max_turns = int(e2e.get("locomo_max_turns", 60))
    lme_num = int(e2e.get("lme_num_cases", 3))
    lme_max_turns = int(e2e.get("lme_max_turns_per_case", 100))

    backend = Mem0Backend(settings)
    store = HugeGraphStore(settings)
    pipeline = EvaluationPipeline(backend, settings, graph_store=store)

    eval_cases = []

    # ---------- Part A: LoCoMo ----------
    print(f"\n[A] LoCoMo conversation #{locomo_idx}（截断 {locomo_max_turns} turns）")
    locomo = LocomoDataset(settings.locomo_path)
    locomo.load()
    # cases 顺序即 conversation 顺序；取第 idx 个 conversation 的全部 QA
    conv_ids: list = []
    for c in locomo.cases:
        cid = c.metadata.get("conversation_id")
        if cid not in conv_ids:
            conv_ids.append(cid)
    target_conv = conv_ids[locomo_idx]
    conv_cases = [c for c in locomo.cases if c.metadata.get("conversation_id") == target_conv]
    case_a = conv_cases[0]
    eval_cases.append(case_a)
    n_added_a = pipeline.ingest_case(case_a, max_turns=locomo_max_turns)
    user_a = pipeline.case_user_id(case_a)
    mem_a = backend.get_all_memories(user_id=user_a)
    print(f"    conversation_id={target_conv}, QA={len(conv_cases)}, 采用 case={case_a.case_id}")
    print(f"    ingest 新增记忆 {n_added_a} 条，当前记忆 {len(mem_a)} 条 (user={user_a})")

    # ---------- Part B: LongMemEval-S ----------
    print(f"\n[B] LongMemEval-S 前 {lme_num} 个 case（每个截断 {lme_max_turns} turns）")
    lme = LongMemEvalDataset(settings.longmemeval_path)
    lme.load()
    for case_b in lme.cases[:lme_num]:
        added = pipeline.ingest_case(case_b, max_turns=lme_max_turns)
        user_b = pipeline.case_user_id(case_b)
        n_b = len(backend.get_all_memories(user_id=user_b))
        print(f"    {case_b.case_id}: +{added} 记忆, 共 {n_b} (type={case_b.category})")
        eval_cases.append(case_b)

    # ---------- Part C: Graph build ----------
    print("\n[C] GraphBuilder: memories → HugeGraph（make_test_schema, reset 重建）")
    all_memories = []
    for c in eval_cases:
        all_memories.extend(backend.get_all_memories(user_id=pipeline.case_user_id(c)))
    builder = GraphBuilder(store)
    report = builder.build(all_memories, make_test_schema(), reset=True)
    edge_total = store.run_gremlin("g.E().count()")["result"]["data"][0]
    print(f"    输入 memories : {len(all_memories)}")
    print(f"    vertices      : {report.vertices_written} (graph count={store.count_vertices('Memory')})")
    print(f"    edges         : {report.edges_written} (graph count={edge_total})")
    print(f"    耗时          : {report.duration_s}s")
    print("    样例 vertex:")
    sample = store.run_gremlin("g.V().hasLabel('Memory').limit(3)")

    def _prop(props: dict, name: str):
        """兼容 gremlin 结果里属性值的两种形态: "v" 或 {"value": "v"}"""
        v = props.get(name)
        return v.get("value") if isinstance(v, dict) else v

    for v in sample["result"]["data"]:
        props = v.get("properties", {})
        print(f"      - {_prop(props, 'memory_id') or v.get('id')}: "
              f"{str(_prop(props, 'content'))[:80]}")
    print("    样例 edge:")
    edges = store.run_gremlin("g.E().hasLabel('RELATED_TO').limit(3)")
    for e in edges["result"]["data"]:
        print(f"      - {e.get('outV')} -[{e.get('label')}]-> {e.get('inV')}")

    # ---------- Part D: 小规模评测对照 ----------
    print("\n[D] Evaluation 对照（graph_enabled=false vs true）")
    results_off = pipeline.run(eval_cases, graph_enabled=False, ingest=False)
    pipeline.print_summary("Mem0 only ", results_off)
    results_on = pipeline.run(eval_cases, graph_enabled=True, ingest=False)
    pipeline.print_summary("Mem0+Graph", results_on)
    for r_off, r_on in zip(results_off, results_on):
        print(f"    {r_off.case_id}: f1 {r_off.metrics['token_f1']} -> {r_on.metrics['token_f1']} | "
              f"contains {r_off.metrics['contains']} -> {r_on.metrics['contains']}")
    print(f"    汇总 graph_off: {summarize(results_off)}")
    print(f"    汇总 graph_on : {summarize(results_on)}")
    print("    注：冒烟用截断数据（e2e.* 可配），gold 事实可能在截断窗口外，指标为 0 属预期；")
    print("       本步骤验证的是链路完整性（检索/rerank/graph 邻居/作答/落盘），不是精度。")
    print(f"    结果已写入: {settings.results_dir}/eval_*_graph{{on,off}}.jsonl")

    print("\n" + BANNER)
    print("✓ Memory → Graph works")
    print("✓ Evaluation pipeline works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
