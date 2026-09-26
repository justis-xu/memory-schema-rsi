#!/usr/bin/env python3
"""Build a fixed, manually labelled source-to-answer sidecar for four cases.

The labels below are human diagnoses. This script only verifies that their
source turns and archived final-context memory IDs still resolve, then copies
the original text so the labels are auditable without rerunning a model.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/analysis/m2_fact_funnel_packet_20260926.json"
OUT = ROOT / "results/analysis/m2_fact_support_sidecar_20260926.json"

SPEC = {
    "locomo_conv-30_qa5": [
        ("临水", ["D1:20"], ["0d8e6624-802d-4167-b7d3-3746bd16c393"], "入槽且作答", "源话、记忆和回答都保留临水"),
        ("自然采光", ["D2:4"], ["9050d972-6347-4473-a97d-ccabe7f6a903"], "入槽但答案遗漏", "第 4 条记忆保留自然采光，回答只写临水"),
        ("马利地胶", ["D2:8"], ["13c9777f-dbd6-41f1-8eec-15802adfb5f4"], "入槽但答案遗漏", "第 6 条记忆保留地胶，回答只写临水"),
    ],
    "locomo_conv-26_qa58": [
        ("昨天做盘子，对应 2023-08-24", ["D14:4"], ["5c43f1b2-8289-4513-9e28-914dedd5ae03"], "提取后时间精度丢失", "源话是昨天，记忆变成最近；会话日只能在保留昨天时用于换算"),
    ],
    "locomo_conv-43_qa41": [
        ("奇幻小说", ["D15:3"], ["3325426c-88b6-46e2-b7bb-d1303bce5de1"], "入槽且作答", "第 1 条记忆和答案都保留奇幻小说"),
        ("情节转折", ["D16:1"], ["983b9769-e399-435b-81f6-b1d94008c683"], "入槽但答案遗漏", "第 6 条记忆保留情节转折，回答未提"),
    ],
    "locomo_conv-43_qa96": [
        ("最爱弹的主题曲属于《哈利·波特与魔法石》", ["D8:14", "D8:15", "D8:16"], ["00e9d949-4f9b-4b1d-8248-96852ed3a6bb", "e5298e7a-052a-41d5-9c67-1da47589696c"], "跨轮关系未显式保留，答案错接", "问答链由 D8:15 的那部电影连接；两条记忆分别保留主题曲和电影，未显式连接，回答猜成星球大战"),
    ],
}


def main() -> None:
    packet = json.loads(PACKET.read_text())
    dataset_path = Path(packet["source_dataset"])
    dataset = {row["sample_id"]: row for row in json.loads(dataset_path.read_text())}
    cases = {row["case_id"]: row for row in packet["cases"]}
    rows = []
    for case_id, atoms in SPEC.items():
        archived = cases[case_id]
        conversation_id = case_id.removeprefix("locomo_").split("_qa")[0]
        source = dataset[conversation_id]
        turns = {
            turn["dia_id"]: (session, source["conversation"][f"{session}_date_time"], turn)
            for session, records in source["conversation"].items()
            if session.startswith("session_") and isinstance(records, list)
            for turn in records
        }
        memories = {memory["id"]: memory for memory in archived["top15_memories"]}
        atom_rows = []
        for claim, source_ids, memory_ids, diagnosis, rationale in atoms:
            cited_turns = []
            for turn_id in source_ids:
                session, session_date, turn = turns[turn_id]
                cited_turns.append({
                    "dia_id": turn_id,
                    "session_id": session,
                    "session_date": session_date,
                    "speaker": turn.get("speaker"),
                    "text": turn.get("text"),
                    "query": turn.get("query"),
                })
            cited_memories = [
                {"id": memory_id, "rank": memories[memory_id]["rank"], "content": memories[memory_id]["content"]}
                for memory_id in memory_ids
            ]
            atom_rows.append({
                "claim": claim,
                "source_turns": cited_turns,
                "final_context_memories": cited_memories,
                "manual_diagnosis": diagnosis,
                "manual_rationale": rationale,
            })
        rows.append({
            "case_id": case_id,
            "question": archived["question"],
            "gold": archived["gold"],
            "archived_answer": archived["answer"],
            "archived_strict_grade": archived["strict_final"],
            "atoms": atom_rows,
        })
    result = {
        "scope": "Four fixed Chinese LoCoMo cases with contrasting source-to-answer failure points",
        "source_packet": str(PACKET.relative_to(ROOT)),
        "source_dataset": str(dataset_path),
        "method": "Manual atom labels; source text, final-context memory IDs and archived answers copied and resolved by script",
        "cases": rows,
        "limits": [
            "Four mechanism cases are not a prevalence sample.",
            "Gold evidence and manual source-turn links are diagnostic oracle data unavailable to the live system.",
            "A memory's archived final-context rank does not prove the answerer used it.",
            "The cross-turn Harry Potter reference is a discourse inference supported by the adjacent question, not an explicit single-turn equation.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT} with {len(rows)} cases and {sum(len(row['atoms']) for row in rows)} claims")


if __name__ == "__main__":
    main()
