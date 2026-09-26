"""LongMemEval-S 适配器单测：cleaned 版（haystack_*）与原版（history/evidence）双格式。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_rsi.benchmarks.longmemeval import LongMemEvalDataset


def make_cleaned_item() -> dict:
    """实测 longmemeval_s_cleaned.json 的真实结构。"""
    return {
        "question_id": "e47becba",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer_session_ids": ["answer_280352e9"],
        "haystack_session_ids": ["sharegpt_yywfIrx_0", "session_0002"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/22 (Mon) 08:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I bought two tennis rackets."},
                {"role": "assistant", "content": "Sounds great!"},
            ],
            [
                {"role": "user", "content": "I got another tennis racket."},
            ],
        ],
    }


def make_original_item() -> dict:
    """原版 LongMemEval-S 结构（history 逐 turn 平铺 + evidence）。"""
    return {
        "qa_id": "q_001",
        "question_type": "multi-session",
        "question": "How many tennis rackets do I own?",
        "answer": "three",
        "question_date": "2023/06/06 (Wed) 07:00",
        "answer_session_ids": ["session_0001", "session_0003"],
        "evidence": [
            {"timestamp": "2022/05/04 (Wed) 06:00", "session_id": "session_0001",
             "role": "user", "content": "I bought two tennis rackets."},
            {"timestamp": "2023/02/01 (Wed) 08:00", "session_id": "session_0003",
             "role": "user", "content": "I got another tennis racket."},
        ],
        "history": [
            {"timestamp": "2022/05/04 (Wed) 06:00", "session_id": "session_0001",
             "role": "user", "content": "I bought two tennis rackets."},
            {"timestamp": "2022/05/04 (Wed) 06:01", "session_id": "session_0001",
             "role": "assistant", "content": "Sounds great!"},
            {"timestamp": "2022/08/10 (Wed) 06:00", "session_id": "session_0002",
             "role": "user", "content": "I will graduate soon."},
            {"timestamp": "2023/02/01 (Wed) 08:00", "session_id": "session_0003",
             "role": "user", "content": "I got another tennis racket."},
        ],
    }


def test_cleaned_format():
    ds = LongMemEvalDataset(data=[make_cleaned_item()])
    assert ds.count() == 1
    case = ds.cases[0]

    assert case.benchmark == "longmemeval_s"
    assert case.case_id == "e47becba"
    assert case.category == "single-session-user"
    assert case.answer == "Business Administration"
    assert case.evidence is None  # cleaned 版无 evidence 字段

    # haystack 三列表一一对应归组
    assert [s["session_id"] for s in case.history] == ["sharegpt_yywfIrx_0", "session_0002"]
    assert case.history[0]["date"] == "2023/05/20 (Sat) 02:21"
    assert [t["role"] for t in case.history[0]["turns"]] == ["user", "assistant"]

    # 原始 record 保留
    assert case.metadata["raw"]["question_id"] == "e47becba"
    assert case.metadata["answer_session_ids"] == ["answer_280352e9"]
    assert case.metadata["question_date"].startswith("2023/05/30")
    assert case.history_stats() == {"sessions": 2, "turns": 3}


def test_original_format():
    ds = LongMemEvalDataset(data=[make_original_item()])
    case = ds.cases[0]

    assert case.case_id == "q_001"
    assert case.category == "multi-session"
    assert len(case.evidence) == 2
    # history 按 session_id 归组，顺序保持
    assert [s["session_id"] for s in case.history] == ["session_0001", "session_0002", "session_0003"]
    assert [t["role"] for t in case.history[0]["turns"]] == ["user", "assistant"]
    assert case.metadata["answer_session_ids"] == ["session_0001", "session_0003"]


def test_stats():
    ds = LongMemEvalDataset(data=[make_cleaned_item()])
    stats = ds.stats()
    assert stats["questions"] == 1
    assert stats["by_question_type"] == {"single-session-user": 1}
    assert stats["sessions"] == 2 and stats["turns"] == 3
