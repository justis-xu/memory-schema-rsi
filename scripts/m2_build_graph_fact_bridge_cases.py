#!/usr/bin/env python3
"""Archive a fixed set of Chinese graph fact-bridge and harm cases.

Case labels are manual. Source turns, contexts, grades and answers are copied
from immutable archived files; no model or mutable graph store is queried.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results_zh/zhfull_locomo_20260925_164645.jsonl"
GRAPH = ROOT / "results_zh/zhfull_fused_locomo_20260925_172126.jsonl"
BASE_GRADE = ROOT / "results/precision_regrade_base.jsonl"
GRAPH_GRADE = ROOT / "results/precision_regrade_graph.jsonl"
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_graph_fact_bridge_cases_20260926.json"

# Selected after inspecting archived flips to compare genuinely missing facts,
# a multimodal source, direct evidence displacement and answer distraction.
CASES = {
    "locomo_conv-47_qa96": {
        "manual_diagnosis": "图补入独有的曼城球迷事实；源文字 D13:15 直接支持；无图 15 条缺该事实",
        "critical_memory_ids": ["3cded448-5856-4da0-81c6-aa9c84cdf61d"],
    },
    "locomo_conv-48_qa39": {
        "manual_diagnosis": "图补入独有的番茄工作法；源文字 D10:4 直接支持；无图答案漏此项",
        "critical_memory_ids": ["8f83084c-1dc8-4bad-b0c8-cb646a24cf17"],
    },
    "locomo_conv-43_qa25": {
        "manual_diagnosis": "图补入约翰的签名篮球；D7:7 图片 caption 和 D7:8-9 对话共同支持，文字单独不点明篮球",
        "critical_memory_ids": ["1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"],
        "extra_source_turn_ids": ["D7:8"],
    },
    "locomo_conv-41_qa86": {
        "manual_diagnosis": "无图第 15 条的汽车抛锚证据被图候选挤出；新增图记忆谈慈善而非修车",
        "critical_memory_ids": ["e9ead35e-3c3b-49ad-8212-86c2e70faa88"],
    },
    "locomo_conv-30_qa80": {
        "manual_diagnosis": "商业计划/投资推介/线上平台的直接记忆仍入槽，但图臂答案转向社媒等其他计划；属答案受干扰病例，非直接证据挤出",
        "critical_memory_ids": ["d4810a3f-ef1f-4778-bd1a-49b2200f7cc8"],
    },
}


def load_jsonl(path):
    return {row["case_id"]: row for row in map(json.loads, path.open())}


def context(row):
    records = {m["id"]: m for m in row["retrieved_memories"] + row["graph_memories"]}
    ids = row["metadata"]["answer_context_ids"]
    graph_ids = set(row["metadata"].get("graph_context_ids") or [])
    assert len(ids) == len(set(ids)) and set(ids) <= records.keys()
    return [{
        "rank": rank, "id": memory_id,
        "source_path": "graph" if memory_id in graph_ids else "vector",
        "content": records[memory_id]["content"],
        "session_date": (records[memory_id].get("metadata") or {}).get("session_date"),
    } for rank, memory_id in enumerate(ids, 1)]


def main():
    base, graph = load_jsonl(BASE), load_jsonl(GRAPH)
    base_grade, graph_grade = load_jsonl(BASE_GRADE), load_jsonl(GRAPH_GRADE)
    dataset = {row["sample_id"]: row for row in json.loads(DATASET.read_text())}
    cases = []
    for case_id, spec in CASES.items():
        b, g = base[case_id], graph[case_id]
        bctx, gctx = context(b), context(g)
        bid, gid = {m["id"] for m in bctx}, {m["id"] for m in gctx}
        assert g["question"] == b["question"] and g["expected_answer"] == b["expected_answer"]
        assert base_grade[case_id]["final"] != graph_grade[case_id]["final"]
        conv_id = case_id.removeprefix("locomo_").split("_qa")[0]
        source = dataset[conv_id]
        qa = next(q for q in source["qa"] if q["question"] == g["question"])
        wanted = set(qa["evidence"] + spec.get("extra_source_turn_ids", []))
        turns = []
        for session, records in source["conversation"].items():
            if not isinstance(records, list):
                continue
            for turn in records:
                if turn.get("dia_id") in wanted:
                    turns.append({
                        "dia_id": turn["dia_id"], "session_id": session,
                        "session_date": source["conversation"][f"{session}_date_time"],
                        "speaker": turn.get("speaker"), "text": turn.get("text"),
                        "query": turn.get("query"), "blip_caption": turn.get("blip_caption"),
                    })
        assert {turn["dia_id"] for turn in turns} == wanted
        assert set(spec["critical_memory_ids"]) <= bid | gid
        cases.append({
            "case_id": case_id, "category": base_grade[case_id]["category"],
            "question": g["question"], "gold": g["expected_answer"],
            "gold_evidence_ids": qa["evidence"], "source_turns": turns,
            "base_grade": base_grade[case_id]["final"],
            "graph_grade": graph_grade[case_id]["final"],
            "base_judge": base_grade[case_id]["judge"],
            "graph_judge": graph_grade[case_id]["judge"],
            "base_answer": b["predicted_answer"],
            "graph_answer": g["predicted_answer"],
            "base_context": bctx, "graph_context": gctx,
            "gained_ids": [m["id"] for m in gctx if m["id"] not in bid],
            "displaced_ids": [m["id"] for m in bctx if m["id"] not in gid],
            "critical_memory_ids": spec["critical_memory_ids"],
            "manual_diagnosis": spec["manual_diagnosis"],
        })
    OUT.write_text(json.dumps({
        "scope": "Five retrospectively selected Chinese paired graph flips, not a prevalence sample or causal effect estimate",
        "base_run": str(BASE.relative_to(ROOT)), "graph_run": str(GRAPH.relative_to(ROOT)),
        "source_dataset": str(DATASET), "cases": cases,
        "limits": [
            "Two independent runs differ in more than graph selection; answer and judge variability remains.",
            "The image-backed signed-ball fact combines caption and dialogue; source text alone is insufficient.",
            "Retrospective selection enriches interpretable examples and cannot estimate frequency among all flips.",
        ],
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT} with {len(cases)} cases")


if __name__ == "__main__":
    main()
