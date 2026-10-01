#!/usr/bin/env python3
"""构建事实来源门离线开发集（A: LongMemEval 工程师 5 条；B: 跨场归并 4 病例×2 候选；C: LoCoMo 角色见证 10 条）。

只读既有归档，零模型调用、零库/图写入。oracle 与门输入物理分离：
gate_inputs[*] 不含任何人工判断字段；oracle 顶层独立键，门实现不得读取。
输出: results/analysis/m2_fact_gate_devset_<date>.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results/analysis/m2_fact_gate_devset_20261001.json"

REVIEW_A = ROOT / "results/analysis/m2_isolated_mem0_ingest_review_20260929.json"
TRACE_A = ROOT / "results/analysis/m2_isolated_mem0_ingest_trace_20260929.jsonl"
ELIGIBILITY_B = ROOT / "results/analysis/m2_carryover_cache_eligibility_20260928.json"
PACKET_B = ROOT / "results/analysis/m2_cross_session_carryover_packet_20260926.json"
SAMPLE_C = ROOT / "results/analysis/m2_role_label_source_sample_20260929.json"
LOCOMO_ZH = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")

B_PAIRS = [
    "conv-43_de89be96dda6",  # 蒂姆写作：迟到事实"写作能创造全新世界"
    "conv-43_c95588df7268",  # 城堡愿望：旧版独有细节（梦想参观城堡、某些作者灵感）
    "conv-26_444350391e64",  # 梅拉妮陶艺：过强情绪（缺"但我还好"限定）
    "conv-41_e94a25c4f01c",  # 公路旅行：两个无源同行者表述
]

ORACLE_KEYS = {
    "support", "reason", "facts", "requires_other_speaker_or_channel",
    "manual_note", "witnesses", "label_corrections", "claim_zh",
    "in_older_memory", "in_later_memory",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_group_a() -> tuple[list[dict], dict]:
    review = json.loads(REVIEW_A.read_text(encoding="utf-8"))
    trace = load_jsonl(TRACE_A)
    batch = next(l["batch"] for l in trace if "batch" in l)
    outcome = next(l for l in trace if "returned_records" in l)
    records = outcome["returned_records"]
    assert len(review["review"]) == 5, len(review["review"])
    assert len(records) == 5, len(records)
    turns = batch["source_turns"]
    assert len(turns) == 12, len(turns)

    gate_inputs, oracle = [], {}
    for i, rev in enumerate(review["review"]):
        rec = next(r for r in records if r["id"] == rev["record_id"])
        gate_inputs.append({
            "input_id": f"A{i}",
            "group": "A",
            "benchmark": "longmemeval",
            "case_note": "LongMemEval 工程师早场真实隔离写入返回记忆",
            "candidate_memory": {
                "memory_id": rec["id"],
                "content": rec["content"],
                "written_session_id": batch["session_id"],
                "written_session_date": batch["session_date"],
            },
            "source_turns": [
                {
                    "turn_index": t["turn_index"],
                    "role": t["role"],
                    "content": t["content"],
                    "source_ref": t.get("source_ref"),
                }
                for t in turns
            ],
        })
        oracle[f"A{i}"] = {
            "record_id": rev["record_id"],
            "support": rev["support"],
            "source_turns": rev["source_turns"],
            "reason": rev["reason"],
        }
    for t in turns:
        assert 0 <= t["turn_index"] < 12
    return gate_inputs, oracle


def build_group_b() -> tuple[list[dict], dict]:
    elig = json.loads(ELIGIBILITY_B.read_text(encoding="utf-8"))
    packet = json.loads(PACKET_B.read_text(encoding="utf-8"))
    inv_pairs = {p["pair_id"]: p for p in elig["corrected_fact_inventory"]["pairs"]}
    pkt_pairs = {p["pair_id"]: p for p in packet["pairs"]}
    corrections = {c["fact_id"]: c for c in elig["label_corrections"]}

    gate_inputs, oracle = [], {}
    for pair_id in B_PAIRS:
        inv = inv_pairs[pair_id]
        pkt = pkt_pairs[pair_id]
        sessions = pkt["source_sessions"]
        session_turns = {}
        for sid, s in sessions.items():
            session_turns[sid] = {
                "date": s["date"],
                "turns": [
                    {
                        "turn_index": j,
                        "speaker": t["speaker"],
                        "content": t["text"],
                        "source_ref": t["dia_id"],
                    }
                    for j, t in enumerate(s["turns"])
                ],
            }
        for slot, mem_key in (("older", "older_memory"), ("later", "later_memory")):
            mem = inv[mem_key]
            input_id = f"B_{slot}_{pair_id}"
            gate_inputs.append({
                "input_id": input_id,
                "group": "B",
                "benchmark": "locomo",
                "case_note": f"跨场归并病例 {pair_id} 的 {slot} 候选",
                "candidate_memory": {
                    "memory_id": mem["id"],
                    "content": mem["content"],
                    "written_session_id": mem["session_id"],
                },
                "source_sessions": session_turns,
            })
            facts = []
            for f in inv["facts"]:
                entry = {
                    "fact_id": f["fact_id"],
                    "claim_zh": f["claim_zh"],
                    "in_older_memory": f["in_older_memory"],
                    "in_later_memory": f["in_later_memory"],
                    "support": f["support"],
                    "source_turns": [
                        {
                            "session_id": st["session_id"],
                            "dia_id": st["dia_id"],
                            "speaker": st["speaker"],
                            "channel": st["channel"],
                            "quote": st["quote"],
                        }
                        for st in f["source_turns"]
                    ],
                }
                if f["fact_id"] in corrections:
                    entry["label_correction"] = corrections[f["fact_id"]]
                facts.append(entry)
            oracle[input_id] = {"pair_id": pair_id, "slot": slot, "facts": facts}

        # oracle dia_id 必须落在对应 session 的真实话轮里
        for f in inv["facts"]:
            for st in f["source_turns"]:
                pool = session_turns[st["session_id"]]["turns"]
                assert any(t["source_ref"] == st["dia_id"] for t in pool), (
                    f"{pair_id} {st['dia_id']} 不在 {st['session_id']}"
                )
    return gate_inputs, oracle


def build_group_c() -> tuple[list[dict], dict]:
    sample = json.loads(SAMPLE_C.read_text(encoding="utf-8"))
    locomo = json.loads(LOCOMO_ZH.read_text(encoding="utf-8"))
    convs = {s["sample_id"]: s for s in locomo}

    gate_inputs, oracle = [], {}
    for i, case in enumerate(sample["cases"]):
        conv_id = case["conversation"]
        session_name = case["written_session"]
        conv = convs[conv_id]["conversation"]
        turns_raw = conv[session_name]
        date_time = conv[f"{session_name}_date_time"]
        turns = []
        for j, t in enumerate(turns_raw):
            turns.append({
                "turn_index": j,
                "speaker": t.get("speaker"),
                "content": t.get("text"),
                "source_ref": t.get("dia_id"),
                "image_query": t.get("query"),
                "blip_caption": t.get("blip_caption"),
            })
        dia_pool = {t["source_ref"] for t in turns}
        for w in case["source_witnesses"]:
            assert w["dia_id"] in dia_pool, f"{conv_id} {w['dia_id']} 不在 {session_name}"
            match = next(t for t in turns if t["source_ref"] == w["dia_id"])
            assert match["content"] == w["text"], f"{conv_id} {w['dia_id']} 正文与数据集不一致"

        input_id = f"C{i}"
        gate_inputs.append({
            "input_id": input_id,
            "group": "C",
            "benchmark": "locomo",
            "case_note": f"LoCoMo {conv_id} assistant 标签记忆（written {session_name}）",
            "candidate_memory": {
                "memory_id": case["memory_id"],
                "content": case["content"],
                "attributed_to": case["attributed_to"],
                "written_session_id": session_name,
                "written_session_date": date_time,
            },
            "source_turns": turns,
        })
        oracle[input_id] = {
            "conversation": conv_id,
            "requires_other_speaker_or_channel": case["requires_other_speaker_or_channel"],
            "witnesses": case["source_witnesses"],
            "manual_note": case["manual_note"],
        }
    return gate_inputs, oracle


def main() -> int:
    a_in, a_or = build_group_a()
    b_in, b_or = build_group_b()
    c_in, c_or = build_group_c()
    gate_inputs = a_in + b_in + c_in
    oracle = {**a_or, **b_or, **c_or}

    # 门输入不得携带任何 oracle 字段（递归检查）
    def walk(node):
        if isinstance(node, dict):
            assert not (ORACLE_KEYS & set(node)), f"门输入泄漏 oracle 字段: {ORACLE_KEYS & set(node)}"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for gi in gate_inputs:
        walk(gi)

    assert len(gate_inputs) == 23, len(gate_inputs)
    counts = {"A": 5, "B": 8, "C": 10}
    for g, n in counts.items():
        got = sum(1 for x in gate_inputs if x["group"] == g)
        assert got == n, (g, got)

    devset = {
        "scope": "事实来源门离线开发集：A=LongMemEval 工程师隔离写入 5 条；B=跨场归并 4 病例×旧/新两候选；C=LoCoMo assistant 标签 10 条。oracle 仅离线验收用。",
        "built_from": {
            str(REVIEW_A.relative_to(ROOT)): sha256_file(REVIEW_A),
            str(TRACE_A.relative_to(ROOT)): sha256_file(TRACE_A),
            str(ELIGIBILITY_B.relative_to(ROOT)): sha256_file(ELIGIBILITY_B),
            str(PACKET_B.relative_to(ROOT)): sha256_file(PACKET_B),
            str(SAMPLE_C.relative_to(ROOT)): sha256_file(SAMPLE_C),
            str(LOCOMO_ZH): sha256_file(LOCOMO_ZH),
        },
        "counts": {"gate_inputs": len(gate_inputs), **counts},
        "oracle_separation": "gate_inputs 不含 ORACLE_KEYS 内任何字段；oracle 键与门输入 input_id 一一对应。",
        "acceptance_cases": [
            {"case": "keep_team_size", "expect": "keep",
             "targets": ["A0"], "detail": "纠正后 4 名工程师+经理共 5 人（turn 10）必须保留"},
            {"case": "keep_late_only_facts", "expect": "keep",
             "targets": ["B_later_conv-43_de89be96dda6"],
             "detail": "迟到旧源事实'写作能创造全新世界/带给蒂姆快乐'必须保留（不得整条撤回）"},
            {"case": "keep_older_only_details", "expect": "keep",
             "targets": ["B_older_conv-43_c95588df7268", "B_older_conv-26_444350391e64"],
             "detail": "旧版独有细节（梦想参观城堡、某些作者灵感、读书画画）不得被判无源删除"},
            {"case": "keep_human_assistant_class", "expect": "keep",
             "targets": [f"C{i}" for i in range(10)],
             "detail": "10 条 assistant 标签记忆全是真人第二说话者，禁止整类拒绝；conv-44 需两真人分别支持"},
            {"case": "block_unsourced_companions", "expect": "withdraw",
             "targets": ["B_older_conv-41_e94a25c4f01c", "B_later_conv-41_e94a25c4f01c"],
             "detail": "无源同行者（'家人'/'玛丽亚'互斥）必须撤回"},
            {"case": "block_assistant_only_clauses", "expect": "withdraw",
             "targets": ["A1", "A3", "A4"],
             "detail": "'含本人'、萨曼莎职位/套餐细节、破冰例子等仅助手来源子句必须撤回或拆出"},
            {"case": "block_overstated_emotion", "expect": "weaken",
             "targets": ["B_older_conv-26_444350391e64", "B_later_conv-26_444350391e64"],
             "detail": "'难受'缺'但我还好'限定（源 D17:10 有原话）须削弱或标注，不得原样放行"},
            {"case": "no_whole_partial_pass", "expect": "split",
             "targets": ["A1", "A3", "A4"],
             "detail": "partial 记忆禁止整条放行，必须拆句分判"},
        ],
        "gate_inputs": gate_inputs,
        "oracle": oracle,
    }

    OUT_PATH.write_text(json.dumps(devset, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ devset 写入 {OUT_PATH.relative_to(ROOT)}")
    print(f"  gate_inputs={len(gate_inputs)} (A5/B8/C10), oracle={len(oracle)}, acceptance_cases=8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
