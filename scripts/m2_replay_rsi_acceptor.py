#!/usr/bin/env python
"""阶段 3：RSI 经验接受器离线重放——事实门 vs 关键词门 vs 全收。

数据：memory-prompt 仓 9 道增益题回源包（m2_rsi_mined_gains_20260928.json，
zh 原话逐 turn）+ 既有关键词门回放（m2_rsi_accept_gate_replay_20260928.json）。
本重放用主仓阶段 1 的事实门（v2 prompt + v3 守卫）做经验接受器：
  经验 = peer 题的 baseline_answer；源 = 该经验被挖掘到的 zh 源 turn（含渠道）。
  接受 ⟺ 门后存在非 withdraw 子句且无待核引用。
对比三种策略各blocked多少经验、多少目标题失去源 ID 增益。

预算：≤15 次 LLM 调用（9 题 × 1-2 经验）。产出
results/analysis/m2_rsi_acceptor_factgate_20261001.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

MP = Path("/Users/xu/git/memory-prompt")
GAINS = MP / "results/analysis/m2_rsi_mined_gains_20260928.json"
KW = MP / "results/analysis/m2_rsi_accept_gate_replay_20260928.json"
OUT = PROJECT_ROOT / "results/analysis/m2_rsi_acceptor_factgate_20261001.json"
LLM_CAP = 15


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"拒绝重跑：{OUT} 已存在")
    gains = json.loads(GAINS.read_text(encoding="utf-8"))
    kw = json.loads(KW.read_text(encoding="utf-8"))
    kw_blocked = set(kw["blocked_peer_ids"])

    from m2_build_independent_graph_zh import Budget, gate_one
    from schema_rsi.config import get_settings
    from schema_rsi.llm.chat import make_chat_client

    settings = get_settings(str(PROJECT_ROOT / "config/locomo_zh_indep.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0, timeout=60)
    budget = Budget()

    rows = []
    for case in gains["cases"]:
        turns = case.get("chinese_source_turns") or {}
        for peer in case.get("peer_paths") or []:
            support_ids = [s["id"].split("|")[-1] for s in peer.get("mined_supports") or []]
            support_turns = [turns[d] for d in support_ids if d in turns]
            if not support_turns:
                rows.append({"case_id": case["id"], "peer_id": peer["id"],
                             "verdict": "no_mined_turn_text", "accepted": False})
                continue
            src_lines = []
            for t in support_turns:
                extra = ""
                if t.get("image_query"):
                    extra += f" [图片query: {t['image_query']}]"
                if t.get("image_caption"):
                    extra += f" [图片描述: {t['image_caption']}]"
                src_lines.append(f"# {t.get('session')} | {t.get('speaker')}: {t['text']}{extra}")
            answer = (peer.get("baseline_answer") or "").replace("**", "").strip()
            gi = {
                "input_id": f"{case['id']}::{peer['id']}", "group": "RSI",
                "benchmark": "locomo",
                "candidate_memory": {"memory_id": peer["id"], "content": answer,
                                     "written_session_id": None, "attributed_to": None},
                "source_turns": [{"turn_index": i, "role": None, "speaker": None,
                                  "content": l, "source_ref": None,
                                  "image_query": None, "blip_caption": None}
                                 for i, l in enumerate(src_lines)],
            }
            valid = {str(i) for i in range(len(src_lines))}
            res = gate_one(client, budget, gi, valid)
            accepted = res["admission_status"] in ("sourced", "weaken")
            rows.append({
                "case_id": case["id"], "peer_id": peer["id"],
                "support_turns": support_ids,
                "admission_status": res["admission_status"],
                "record_action": res["record_action"],
                "withdrawn_qualifiers": res["withdrawn_qualifiers"],
                "accepted": accepted,
                "kw_gate_blocked": peer["id"] in kw_blocked,
            })
            print(f"  {case['id']} :: {peer['id']} → {res['admission_status']}"
                  f"（kw门={'blocked' if peer['id'] in kw_blocked else 'pass'}）", flush=True)
            assert budget.llm <= LLM_CAP

    n = len(rows)
    fact_blocked = [r["peer_id"] for r in rows if not r["accepted"]]
    summary = {
        "experiences": n,
        "all_accept_blocked": 0,
        "kw_gate_blocked": sum(1 for r in rows if r["kw_gate_blocked"]),
        "fact_gate_blocked": len(fact_blocked),
        "fact_gate_blocked_ids": fact_blocked,
        "agreement_with_kw_gate": sum(1 for r in rows
                                      if r["accepted"] != r["kw_gate_blocked"]),
        "note": "kw 门按关键词覆盖；事实门按子句语义来源。语义接受但结构 partial 的经验标注 weaken。",
    }
    out = {"scope": "RSI 经验接受器离线重放：事实门(v2+v3) vs 关键词门 vs 全收。",
           "inputs": {"gains": str(GAINS), "kw_replay": str(KW)},
           "llm_calls": budget.llm, "llm_cap": LLM_CAP,
           "summary": summary, "rows": rows}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ 经验 {n}：事实门拒 {len(fact_blocked)}，kw门拒 {summary['kw_gate_blocked']}，"
          f"不一致 {summary['agreement_with_kw_gate']} → {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
