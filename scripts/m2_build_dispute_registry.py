#!/usr/bin/env python
"""阶段 4：争议题规则集 + 不可恢复项固定台账（零模型调用）。

把散在各文档的争议题清单整合为机器可读登记表（题 ID、争议类型、来源文档、
后续实验预注册处理规则），并把"不可恢复项"逐条固定成台账（杜绝反复猜测）。

产出: results/analysis/m2_dispute_registry_20261001.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results/analysis/m2_dispute_registry_20261001.json"

# 来源：docs/m2_zh_translation_semantic_sample_zh.md（6 题）等；每条带来源指针。
DOC_DISPUTES = {
    "gold_evidence_unreachable": (
        ["locomo_conv-26_qa37", "locomo_conv-42_qa58", "locomo_conv-42_qa88",
         "locomo_conv-43_qa18", "locomo_conv-47_qa38", "locomo_conv-49_qa31",
         "locomo_conv-49_qa38", "locomo_conv-49_qa46", "locomo_conv-50_qa69"],
        "results/analysis/gold_evidence_id_audit_20260926.json",
        "evidence 引用不可达源 turn；覆盖类指标须先排除或人工核源"),
    "gold_semantic_mismatch": (
        ["locomo_conv-43_qa56", "locomo_conv-43_qa60"],
        "results/analysis/gold_evidence_quality_zh.md",
        "gold evidence 结构在但语义错接（钢琴→小提琴；D23:2 vs D23:3）；不得用于经验训练或 oracle 超边"),
    "translation_semantic": (
        ["locomo_conv-50_qa13", "locomo_conv-48_qa20", "locomo_conv-26_qa111",
         "locomo_conv-44_qa15", "locomo_conv-48_qa36", "locomo_conv-42_qa215"],
        "docs/m2_zh_translation_semantic_sample_zh.md",
        "翻译语义偏移（如'享受→愿意'）；判分争议须按题面预注册规则处理"),
    "flip_source_ambiguous": (
        ["locomo_conv-43_qa166", "locomo_conv-47_qa61", "locomo_conv-30_qa69", "locomo_conv-42_qa62"],
        "docs/m2_graph_flip_source_sample_zh.md",
        "翻分归因歧义（金标人物错接/题面年份与源不一致/同义判分）"),
}

UNRECOVERABLE = [
    {"item": "旧运行 45 道中文缺题的失败日志",
     "why": "本机归档无对应结构化失败日志，缺题原因不可复原",
     "source": "docs/m2_missing_case_score_bounds_zh.md",
     "action": "固定记为不可恢复；后续运行必须自带失败记账"},
    {"item": "Jev 首判完整有序上下文与服务实际 state",
     "why": "历史运行未归档首判前后全文、候选与 state，同次反事实不可重建",
     "source": "docs/m2_jev_expansion_zh.md",
     "action": "不再从旧档重建；未来运行启用 jev_trace 留痕字段"},
    {"item": "历史提取调用的完整输入输出日志",
     "why": "无法定位旧图候选漂移发生于提示遵循/检索上下文/批次保存哪一层",
     "source": "docs/m2_carryover_fact_merge_zh.md",
     "action": "固定记为不可恢复；新增运行按 source_trace 契约留痕"},
    {"item": "mem0 add 返回层 session_id/session_date",
     "why": "返回结构与库存查询结构不同造成的返回层来源字段缺口（结构限制，非数据丢失）",
     "source": "docs/m2_isolated_mem0_ingest_zh.md",
     "action": "以库存查询为准；修复返回层时须验证 ID 与库存元数据一致"},
    {"item": "旧缓存的内容/模型/提示指纹",
     "why": "命中无法证明提取成功或来源可信",
     "source": "docs/m2_current_cache_graph_readiness_zh.md",
     "action": "旧缓存不得直接发布为当前可靠图；新图必须带 manifest 指纹"},
    {"item": "跨运行缓存可见性的当时真实请求",
     "why": "delete_all 清向量不清历史消息的输入路径已隔离回放，但四条未来事实错挂缺当时请求，不能断言单一成因",
     "source": "https://github.com/justis-xu/memory-prompt/blob/master/docs/m2_cross_run_cache_visibility_zh.md",
     "action": "标记多成因可能；未来运行先清 owner 历史消息"},
    {"item": "Jev GPU 决策服务（laya/decider）",
     "why": "2026-10-01 用户确认实例失效；seetacloud 域名随重开变化",
     "source": "本阶段运行记录",
     "action": "Jev 线停车；恢复后先探测域名再续"},
]


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"拒绝重跑：{OUT} 已存在")
    bounds = json.loads((PROJECT_ROOT / "results/analysis/m2_missing_case_score_bounds_20260929.json")
                        .read_text(encoding="utf-8"))
    missing_groups = [{
        "group": c["group"], "a": Path(c["a"]).name, "b": Path(c["b"]).name,
        "missing_from_either": c["missing_from_either"],
        "sign_robust_to_missing_only": c["sign_robust_to_missing_only"],
    } for c in bounds["comparisons"]]

    disputes = []
    for tag, (ids, source, rule) in DOC_DISPUTES.items():
        for cid in ids:
            disputes.append({"case_id": cid, "tag": tag, "source": source, "rule": rule})

    registry = {
        "scope": "争议题预注册登记表 + 不可恢复项固定台账。后续任何配对实验必须消费本表。",
        "usage_rules": [
            "配对选题默认排除 disputes 内全部题；确需使用时按 rule 预注册处理并在结果中标注。",
            "unrecoverable 项不得反复尝试复原；新增运行必须自带失败记账与留痕。",
        ],
        "disputes": disputes,
        "dispute_counts": {t: sum(1 for d in disputes if d["tag"] == t) for t in DOC_DISPUTES},
        "missing_case_groups": missing_groups,
        "unrecoverable": UNRECOVERABLE,
    }
    OUT.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ 争议 {len(disputes)} 题（{registry['dispute_counts']}），"
          f"不可恢复项 {len(UNRECOVERABLE)}，缺题组 {len(missing_groups)} → {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
