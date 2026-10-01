#!/usr/bin/env python
"""阶段 2b：图配对评测预冻结选题（A 组"图应补当前缺失" / B 组"直证不得挤出"）。

只读 + 每题 1 次向量检索 + 1 次 rerank（零 LLM 提取调用，不计 2a 预算）：
  1. A 组候选：57 图正例翻分题（graph_context_audit 的 new_to_baseline 候选）对
     当前无图 top-15 检索探针——图换入 ID 不在当前 top-15 = "图会改变当前上下文"。
  2. B 组候选：4 个答题证据病例 + 同上下文翻分题（same_ids_content_date），
     must_not_evict = 当前 top-15 中证据场次链接的记忆。
  3. 争议题按预注册清单排除（gold 9+2 / 翻译 6 / 翻分源审 4）。
  4. 冻结包：每题当前 top-15 逐条内容+指纹（sha256），供 2c 冻结复现。

产出: results/analysis/m2_graph_eval_frozen_questions_20261001.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "results/analysis/m2_graph_eval_frozen_questions_20261001.json"
AUDIT = PROJECT_ROOT / "results/analysis/graph_context_audit_20260926.json"
MECH = PROJECT_ROOT / "results/analysis/m2_graph_positive_mechanisms_20260927.json"
LABELS = PROJECT_ROOT / "results/analysis/m2_graph_positive_fact_gain_labels_20260928.json"

DISPUTES = {
    "gold_evidence_unreachable": [
        "locomo_conv-26_qa37", "locomo_conv-42_qa58", "locomo_conv-42_qa88",
        "locomo_conv-43_qa18", "locomo_conv-47_qa38", "locomo_conv-49_qa31",
        "locomo_conv-49_qa38", "locomo_conv-49_qa46", "locomo_conv-50_qa69"],
    "gold_semantic_mismatch": ["locomo_conv-43_qa56", "locomo_conv-43_qa60"],
    "translation_semantic": [
        "locomo_conv-50_qa13", "locomo_conv-48_qa20", "locomo_conv-26_qa111",
        "locomo_conv-44_qa15", "locomo_conv-48_qa36", "locomo_conv-42_qa215"],
    "flip_source_ambiguous": [
        "locomo_conv-43_qa166", "locomo_conv-47_qa61", "locomo_conv-30_qa69",
        "locomo_conv-42_qa62"],
}
B_EVIDENCE_CASES = [
    "locomo_conv-43_qa152", "locomo_conv-50_qa61",
    "locomo_conv-48_qa90", "locomo_conv-41_qa86",
]
TOP_K_VECTOR = 45
TOP_K_FINAL = 15


def content_sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def evidence_sessions(evidence: list[str] | None) -> set[str]:
    out = set()
    for e in evidence or []:
        if isinstance(e, str) and e.startswith("D"):
            num = e.split(":")[0][1:]
            if num.isdigit():
                out.add(f"session_{num}")
    return out


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"拒绝重跑：{OUT} 已存在")
    from schema_rsi.benchmarks.locomo import LocomoDataset
    from schema_rsi.config import get_settings
    from schema_rsi.llm.rerank import RerankClient
    from schema_rsi.memory import Mem0Backend

    settings = get_settings(str(PROJECT_ROOT / "config/locomo_zh.yaml"))
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    by_case = {c.case_id: c for c in ds.cases}

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_cases = {c["case_id"]: c for c in audit["cases"]}
    mech = json.loads(MECH.read_text(encoding="utf-8"))
    eligible = mech["eligible_case_ids"]
    labels = {l["case_id"]: l for l in json.loads(
        LABELS.read_text(encoding="utf-8"))["labels"]}

    dispute_of = {cid: tag for tag, ids in DISPUTES.items() for cid in ids}

    backend = Mem0Backend(settings)
    reranker = RerankClient(settings.rerank.base_url, settings.rerank.api_key, settings.rerank.model)

    all_ids = list(dict.fromkeys(eligible + B_EVIDENCE_CASES))
    probes = {}
    for cid in all_ids:
        case = by_case.get(cid)
        if not case:
            print(f"  !! 数据集缺 {cid}")
            continue
        uid = f"zhfull:locomoo:{cid.split('_')[1]}" if False else f"zhfull:locomo:{cid.removeprefix('locomo_').split('_qa')[0]}"
        vector = backend.search_memory(case.question, user_id=uid, top_k=TOP_K_VECTOR)
        scored = reranker.rerank(case.question, [m.content for m in vector], top_n=TOP_K_FINAL)
        final = [vector[i["index"]] for i in scored if 0 <= i["index"] < len(vector)][:TOP_K_FINAL]
        probes[cid] = [{
            "id": m.id, "content_sha": content_sha(m.content),
            "content": m.content, "session_id": m.metadata.get("session_id"),
        } for m in final]
        print(f"  probe {cid}: top15 就绪", flush=True)

    def in_top15(cid: str, ids: set[str]) -> set[str]:
        return {p["id"] for p in probes.get(cid, [])} & ids

    # ---- A 组 ----
    a_rows = []
    for cid in eligible:
        au = audit_cases.get(cid)
        if not au or cid not in probes:
            continue
        new_ids = set(au.get("new_to_baseline_candidate_ids") or [])
        missing = new_ids - in_top15(cid, new_ids)
        if not missing:
            continue
        lab = labels.get(cid, {})
        a_rows.append({
            "case_id": cid, "group": "A",
            "question": by_case[cid].question,
            "gold": by_case[cid].answer,
            "graph_swapin_missing_from_current": sorted(missing),
            "labeled_graph_new_necessary_fact": bool(lab.get("graph_new_necessary_fact")),
            "label_basis": lab.get("basis"),
            "old_outcome": au.get("outcome"),
            "dispute": dispute_of.get(cid),
        })
    # A 组换入记忆正文从本探针不可得 → 2c 图臂现取时记录

    # ---- B 组 ----
    b_rows = []
    b_pool = list(dict.fromkeys(B_EVIDENCE_CASES + [
        c["case_id"] for c in audit["cases"]
        if c.get("same_ids_content_date") and c.get("outcome") not in ("both_exact", "both_wrong")
    ]))
    for cid in b_pool:
        if cid in probes:
            au = audit_cases.get(cid, {})
            ev_sessions = evidence_sessions(by_case[cid].evidence)
            mne = sorted({p["id"] for p in probes[cid] if p.get("session_id") in ev_sessions})
            if cid not in B_EVIDENCE_CASES and not mne:
                continue  # 同上下文翻分无证据链接 → 不入 B 组
            b_rows.append({
                "case_id": cid, "group": "B",
                "question": by_case[cid].question,
                "gold": by_case[cid].answer,
                "must_not_evict": mne,
                "evidence_sessions": sorted(ev_sessions),
                "old_outcome": au.get("outcome"),
                "dispute": dispute_of.get(cid),
                "is_named_evidence_case": cid in B_EVIDENCE_CASES,
            })

    # ---- 选取 ----
    a_clean = [r for r in a_rows if not r["dispute"]]
    a_sel = sorted(a_clean, key=lambda r: (not r["labeled_graph_new_necessary_fact"],
                                           -len(r["graph_swapin_missing_from_current"])))[:10]
    b_clean = [r for r in b_rows if not r["dispute"]]
    b_named = [r for r in b_clean if r["is_named_evidence_case"]]
    b_rest = sorted([r for r in b_clean if not r["is_named_evidence_case"]],
                    key=lambda r: -len(r["must_not_evict"]))[:10 - len(b_named)]
    b_sel = b_named + b_rest
    selected = a_sel + b_sel
    for r in selected:
        r["current_top15_frozen"] = probes[r["case_id"]]

    pack = {
        "scope": "图配对评测预冻结选题包：A=图换入缺失于当前 top-15（≤10），B=直证不得挤出（≤10）。",
        "method": "当前检索探针（向量45→rerank 15）；A=57 正例换入 ID 缺失；B=4 证据病例+同上下文翻分证据链接。",
        "dispute_excluded": {
            cid: dispute_of[cid] for cid in dispute_of
            if cid in {r["case_id"] for r in a_rows + b_rows}},
        "a_candidate_count": len(a_rows), "b_candidate_count": len(b_rows),
        "selected_count": len(selected),
        "models": {"embedding": settings.embedding.model, "rerank": settings.rerank.model},
        "frozen_questions": selected,
    }
    pack["pack_sha256"] = hashlib.sha256(
        json.dumps(selected, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    OUT.write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ A组 {len(a_sel)}/{len(a_clean)} 候选，B组 {len(b_sel)}/{len(b_clean)} 候选"
          f"（争议排除 {len(pack['dispute_excluded'])}）→ {OUT.name}")
    for r in selected:
        print(f"  [{r['group']}] {r['case_id']} dispute={r['dispute'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
