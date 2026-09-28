"""Audit a small fixed sample of actual Chinese memory outputs against source turns.

The claim labels are human judgments. The code only verifies that selected
records and cited source turns are present in the current local snapshots.
"""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUTPUT = ROOT / "results/analysis/m2_output_origin_sample_20260928.json"
CONVS = ("conv-26", "conv-41", "conv-43", "conv-47")

# Fixed after inspecting the first two hash-sorted records in each conversation.
# (fact, source session, source dia_ids, support of the *written wording*, note)
CLAIMS = {
    "ef55dd2f-2fb7-4d79-a161-13f8f8507f51": [
        ("朋友去年收养，过程漫长，如今一家幸福", "session_17", ["D17:4"], "full", "梅拉妮本人叙述"),
        ("梅拉妮因此考虑自己也收养", "session_17", ["D17:4"], "full", "保留也许的计划模态"),
    ],
    "f87df58f-81e4-46e4-a2bc-22e62b98c625": [
        ("公路旅行期间儿子遭遇事故但平安", "session_18", ["D18:1", "D18:3"], "full", "原话为事故，未明确事故类型"),
        ("事故是车祸", "session_18", ["D18:1"], "partial", "原话为事故，图片说明为车内仪表盘；不能单凭它确定事故类型"),
        ("事故发生在2023年10月20日", "session_18", ["D18:1"], "absent", "10月20日是回忆事故的会话日；原话说上周末"),
        ("梅拉妮因此更珍惜家人", "session_18", ["D18:3"], "full", "本人明确自述"),
        ("孩子们起初害怕但表现坚强", "session_18", ["D18:7"], "full", "本人明确自述"),
    ],
    "0395f295-aa1f-4460-a899-7fa0fa1a7447": [
        ("玛丽亚童年与家人自驾游去俄勒冈州", "session_18", ["D18:3"], "full", "本人明确自述"),
        ("分享的照片显示有人站在峡谷悬崖边", "session_18", ["D18:3"], "full", "图片说明仅称一个人；记忆不能进一步推断这个人就是玛丽亚"),
    ],
    "3e495438-a164-4250-ab0f-9f479a67a7dc": [
        ("约翰在当地学校做志愿导师", "session_31", ["D31:1"], "full", "本人明确自述"),
        ("学生自信和技能提高", "session_31", ["D31:9"], "full", "本人明确自述"),
        ("上周一名学生兴奋地展示作文，约翰自豪", "session_31", ["D31:9"], "full", "相对时间以8月13日源会话为锚"),
    ],
    "0203a396-023b-4604-816d-e13a497c55a3": [
        ("约翰球队最近与强队比赛并获胜", "session_22", ["D22:4"], "full", "本人明确自述"),
        ("全队乘消防车庆祝胜利", "session_22", ["D22:4"], "partial", "图片说明只说一群人乘消防车；没有全队身份或庆祝获胜的文字证据"),
        ("他最看重球场内外建立的情谊", "session_22", ["D22:6"], "full", "本人明确自述"),
    ],
    "b0712fd1-0b3a-47a9-aa5b-d2c4f206c8ef": [
        ("约翰最近没太多时间读书，聊后开始读一本书，感觉好", "session_14", ["D14:23"], "full", "本人明确自述"),
    ],
    "c5ee46ff-cc3b-4a19-abe9-9fe607f99161": [
        ("詹姆斯通常用语音聊天与团队沟通，觉得快且高效", "session_4", ["D4:14"], "full", "本人明确自述"),
    ],
    "47ad2f43-3ea1-474e-9e61-d99350de95d8": [
        ("詹姆斯开始尝试RPG与策略游戏并感到兴奋", "session_3", ["D3:18"], "full", "本人明确自述"),
        ("他答应向约翰更新探索进展", "session_3", ["D3:22"], "full", "本人明确自述"),
    ],
}

# Same core topic already had a record under its source session. These records
# are inspected as context, not included in the eight-record support counts.
EARLIER_CARRIERS = {
    "f87df58f-81e4-46e4-a2bc-22e62b98c625": ["7da557c9-eb9c-4392-b9c5-2ac9863d2695", "2ec90c89-e8d4-4fd1-80bd-2eca86fb2c2f"],
    "3e495438-a164-4250-ab0f-9f479a67a7dc": ["21519f92-6587-4616-926b-120da1779306", "0467de53-751f-4955-9bf3-5b19ded453cc"],
}


def load_memories():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sql = """SELECT e.embedding_id,
      MAX(CASE WHEN m.key='data' THEN m.string_value END) AS content,
      MAX(CASE WHEN m.key='user_id' THEN m.string_value END) AS owner,
      MAX(CASE WHEN m.key='session_id' THEN m.string_value END) AS session_id,
      MAX(CASE WHEN m.key='session_date' THEN m.string_value END) AS session_date,
      MAX(CASE WHEN m.key='attributed_to' THEN m.string_value END) AS attributed_to
      FROM embeddings e JOIN embedding_metadata m ON e.id=m.id GROUP BY e.id"""
    try:
        return [dict(row) for row in conn.execute(sql)]
    finally:
        conn.close()


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    db_hash = hashlib.sha256(DB.read_bytes()).hexdigest()
    sources = {case["sample_id"]: case["conversation"] for case in json.loads(SOURCE.read_text())}
    memories = load_memories()
    by_id = {memory["embedding_id"]: memory for memory in memories}
    selected = []
    for conv in CONVS:
        rows = [m for m in memories if m["owner"] == f"zhfull:locomo:{conv}" and len(m["content"] or "") > 25]
        rows.sort(key=lambda m: hashlib.sha256(m["embedding_id"].encode()).hexdigest())
        selected.extend((conv, row) for row in rows[:2])
    assert {m["embedding_id"] for _, m in selected} == set(CLAIMS), "Snapshot or selection changed"

    output = []
    for conv, memory in selected:
        c = sources[conv]
        rows = []
        for fact, source_session, refs, support, note in CLAIMS[memory["embedding_id"]]:
            witness = []
            for dia_id in refs:
                turn, = [t for t in c[source_session] if t["dia_id"] == dia_id]
                witness.append({"session_id": source_session, "dia_id": dia_id, "speaker": turn["speaker"],
                                "text": turn["text"], "blip_caption": turn.get("blip_caption")})
            rows.append({"fact_zh": fact, "support": support, "note": note, "source_turns": witness})
        origins = sorted({w["session_id"] for row in rows for w in row["source_turns"]})
        assert len(origins) == 1
        carriers = [by_id[mid] for mid in EARLIER_CARRIERS.get(memory["embedding_id"], [])]
        assert all(m["owner"] == memory["owner"] and m["session_id"] == origins[0] for m in carriers)
        output.append({"conversation": conv, "memory": memory, "written_session_date": c[memory["session_id"] + "_date_time"],
                       "supported_origin_session": origins[0], "origin_session_date": c[origins[0] + "_date_time"],
                       "origin_matches_written_session": origins[0] == memory["session_id"], "claims": rows,
                       "same_topic_source_session_records": carriers,
                       "written_session_turns": [{"dia_id": t["dia_id"], "speaker": t["speaker"], "text": t["text"],
                                                  "blip_caption": t.get("blip_caption")}
                                                 for t in c[memory["session_id"]]]
                       if origins[0] != memory["session_id"] else None})
    counts = {"memories": len(output), "claims": sum(len(x["claims"]) for x in output),
              "written_origin_mismatch": sum(not x["origin_matches_written_session"] for x in output),
              "full": sum(c["support"] == "full" for x in output for c in x["claims"]),
              "partial": sum(c["support"] == "partial" for x in output for c in x["claims"]),
              "absent": sum(c["support"] == "absent" for x in output for c in x["claims"])}
    mismatched_ids = {x["memory"]["embedding_id"] for x in output if not x["origin_matches_written_session"]}
    exposure = {mid: {"by_run": {}, "unique_case_ids": []} for mid in mismatched_ids}
    unique = {mid: set() for mid in mismatched_ids}
    for run in sorted((ROOT / "results_zh").glob("zhfull*jsonl")):
        counts_by_id = Counter()
        for line in run.open():
            row = json.loads(line)
            for mid in set(row.get("metadata", {}).get("answer_context_ids", [])) & mismatched_ids:
                candidates = row.get("retrieved_memories", []) + row.get("graph_memories", [])
                matched = [m for m in candidates if m.get("id") == mid]
                assert matched, (run.name, row["case_id"], mid)
                assert all(m["content"] == by_id[mid]["content"] and
                           m["metadata"].get("session_id") == by_id[mid]["session_id"] and
                           m["metadata"].get("session_date") == by_id[mid]["session_date"]
                           for m in matched), (run.name, row["case_id"], mid, "archived memory drift")
                counts_by_id[mid] += 1
                unique[mid].add(row["case_id"])
        for mid in mismatched_ids:
            exposure[mid]["by_run"][run.name] = counts_by_id[mid]
    for mid in mismatched_ids:
        exposure[mid]["unique_case_ids"] = sorted(unique[mid])
    OUTPUT.write_text(json.dumps({"scope": "Current Chinese Chroma outputs; first two SHA256(ID)-sorted records with >25 characters from each of four chosen conversations. Human claim labels; diagnostic sample only.",
                                  "source_path": str(SOURCE), "source_sha256": source_hash, "db_path": str(DB),
                                  "db_sha256": db_hash, "counts": counts, "items": output,
                                  "mismatched_output_answer_context_exposure": exposure}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
