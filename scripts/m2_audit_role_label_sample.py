#!/usr/bin/env python3
"""Fixed one-per-owner source sample of current assistant-attributed LoCoMo memories."""

import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUT = ROOT / "results/analysis/m2_role_label_source_sample_20260929.json"
SEED = "m2_role_sample_20260929|"

# Manually read against the fixed selected records. These are source witnesses,
# not an automatic semantic-support verifier.
WITNESS_IDS = {
    "conv-26": ["D1:2", "D1:18"],
    "conv-30": ["D5:12", "D5:13", "D5:15"],
    "conv-41": ["D19:23"],
    "conv-42": ["D3:4", "D3:6", "D3:8"],
    "conv-43": ["D1:7", "D1:9"],
    "conv-44": ["D20:37", "D20:38", "D20:39", "D20:40"],
    "conv-47": ["D20:9", "D20:10", "D20:11"],
    "conv-48": ["D6:1"],
    "conv-49": ["D17:5"],
    "conv-50": ["D19:1", "D19:3"],
}
MANUAL = {
    "conv-26": "梅拉妮本人：工作孩子忙、带孩子游泳；对话对象姓名来自会话角色。",
    "conv-30": "吉娜本人说纹身含义；具体英文名言来自乔恩上一轮图片query，属跨说话者且跨渠道组合。",
    "conv-41": "玛丽亚本人说佛罗里达度假和与家人感激。",
    "conv-42": "内特本人说尝试椰奶冰淇淋、口味和未来配料。",
    "conv-43": "约翰本人说下周揭幕、目标命中率和训练。",
    "conv-44": "安德鲁承诺带路，奥黛丽回应期待；“两人都期待”需要两位真人话轮。",
    "conv-47": "约翰本人说买设备、翻新桌面，并确认自己有类似椅子键盘和新电脑耳机。",
    "conv-48": "乔琳本人说昨晚聚餐饮酒放松；朋友举杯/食物/酒杯来自该轮图片query/caption，红酒颜色未由文字辅助字段直接支持。",
    "conv-49": "萨姆本人说电影治愈心情；《教父》DVD名称仅由该轮图片query给出。",
    "conv-50": "戴夫本人说即兴演奏开心且因投入而忘记录音。",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit("refusing to overwrite sample")
    entries = {x["sample_id"]: x["conversation"] for x in json.loads(DATASET.read_text())}
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT e.embedding_id,
          MAX(CASE WHEN m.key='data' THEN m.string_value END),
          MAX(CASE WHEN m.key='user_id' THEN m.string_value END),
          MAX(CASE WHEN m.key='session_id' THEN m.string_value END),
          MAX(CASE WHEN m.key='attributed_to' THEN m.string_value END)
        FROM embeddings e JOIN embedding_metadata m ON e.id=m.id GROUP BY e.id
    """).fetchall()
    conn.close()
    by_owner = defaultdict(list)
    for row in rows:
        if row[2] and row[2].startswith("zhfull:locomo:") and row[4] == "assistant":
            by_owner[row[2]].append(row)
    assert len(by_owner) == 10
    cases = []
    for owner, candidates in sorted(by_owner.items()):
        row = min(candidates, key=lambda r: hashlib.sha256((SEED + r[0]).encode()).hexdigest())
        conv_id = owner.rsplit(":", 1)[1]
        conv = entries[conv_id]
        source = {turn["dia_id"]: turn for turn in conv[row[3]]}
        witnesses = []
        for dia_id in WITNESS_IDS[conv_id]:
            turn = source[dia_id]
            witnesses.append({"dia_id": dia_id, "speaker": turn["speaker"],
                              "role_in_adapter": "user" if turn["speaker"] == conv["speaker_a"] else "assistant",
                              "text": turn["text"], "image_query": turn.get("query"),
                              "blip_caption": turn.get("blip_caption")})
        cases.append({"conversation": conv_id, "memory_id": row[0], "content": row[1],
                      "written_session": row[3], "attributed_to": row[4],
                      "speaker_a": conv["speaker_a"], "speaker_b": conv["speaker_b"],
                      "source_witnesses": witnesses, "manual_note": MANUAL[conv_id],
                      "requires_other_speaker_or_channel": conv_id in {"conv-30", "conv-44", "conv-48", "conv-49"}})
    result = {
        "scope": "当前中文LoCoMo每个owner固定哈希抽一条assistant标签记忆，人工回源说话者与渠道",
        "selection": {"seed": SEED, "one_per_owner": True,
                      "candidate_count": sum(map(len, by_owner.values())), "selected_count": len(cases)},
        "input_sha256": {"dataset": sha(DATASET), "chroma_sqlite": sha(DB)},
        "cases": cases,
        "model_calls": 0,
        "limits": ["事后人工选择见证turn；标签不是全句独立语义蕴含自动判断。",
                   "每个owner只取一条，不能估689条的混写比例。",
                   "其它说话者或图片query参与事实，不等于该条整句无源。"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected": len(cases), "with_other_speaker_or_channel":
                      sum(c["requires_other_speaker_or_channel"] for c in cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
