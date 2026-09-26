"""LoCoMo 适配器单测（mini fixture，不依赖真实数据文件）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_rsi.benchmarks.locomo import LocomoDataset


def make_entry() -> dict:
    """mini fixture，结构与官方 locomo10.json 一致（conversation 为嵌套 dict，category 为 int）。"""
    return {
        "sample_id": "conv-99",
        "conversation": {
            "speaker_a": "Lisa",
            "speaker_b": "Ben",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Lisa", "text": "I just moved to Hangzhou."},
                {"dia_id": "D1:2", "speaker": "Ben", "text": "Cool!",
                 "blip_caption": "a cat photo", "query": "my cat"},
            ],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_2": [
                {"dia_id": "D2:1", "speaker": "Lisa", "text": "I have a cat named Milo."},
            ],
            "session_2_date_time": "10:00 am on 15 May, 2023",
            # 只有 date 没有 turn 列表的 session（真实数据常见）应被跳过
            "session_3_date_time": "10:00 am on 20 May, 2023",
        },
        "qa": [
            {
                "question": "Where does Lisa live?",
                "answer": "Hangzhou",
                "category": 1,
                "evidence": ["D1:1"],
            },
            {
                "question": "What pets does Lisa have?",
                "answer": "a cat named Milo",
                "category": 2,
                "evidence": ["D1:2", "D2:1"],
            },
        ],
    }


def test_parse_and_unified_case():
    ds = LocomoDataset(data=[make_entry()])
    assert ds.count() == 2
    case = ds.cases[0]

    # 统一字段
    assert case.benchmark == "locomo"
    assert case.case_id == "locomo_conv-99_qa0"
    assert case.question == "Where does Lisa live?"
    assert case.answer == "Hangzhou"
    assert case.category == "multi-hop"  # int 1 -> 官方名称（multi-hop）
    assert case.evidence == ["D1:1"]

    # history: 2 个有内容的 session（date-only 的 session_3 被跳过），日期保留
    assert len(case.history) == 2
    assert case.history[0]["session_id"] == "session_1"
    assert case.history[0]["date"] == "1:56 pm on 8 May, 2023"
    t0 = case.history[0]["turns"][0]
    assert t0["role"] == "user" and t0["speaker"] == "Lisa"
    assert t0["dia_id"] == "D1:1"
    # 多模态字段折叠进文本
    t1 = case.history[0]["turns"][1]
    assert "shared image" in t1["content"] and "my cat" in t1["content"]

    # 原始 record 保留
    assert case.metadata["raw"]["sample_id"] == "conv-99"
    assert case.metadata["qa_raw"]["category"] == 1

    # history_turns 截断
    assert len(case.history_turns(max_turns=2)) == 2
    assert case.history_stats() == {"sessions": 2, "turns": 3}


def test_stats():
    ds = LocomoDataset(data=[make_entry()])
    stats = ds.stats()
    assert stats["conversations"] == 1
    assert stats["qa_total"] == 2
    assert stats["qa_by_category"] == {"multi-hop": 1, "temporal": 1}
