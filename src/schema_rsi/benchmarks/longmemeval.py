"""LongMemEval-S（原版小版 cleaned，非 V2）数据集适配器。

实测 longmemeval_s_cleaned.json 结构：list of
  {
    "question_id": "e47becba",
    "question_type": single-session-user | single-session-assistant | single-session-preference |
                     multi-session | temporal-reasoning | knowledge-update,
    "question": str, "answer": str, "question_date": str,
    "answer_session_ids": [str, ...],
    "haystack_session_ids": [str, ...],     # 与下面两个列表一一对应
    "haystack_dates": [str, ...],
    "haystack_sessions": [[{"role","content"}, ...], ...]   # 每个 session 是 turn 列表
  }
（此 cleaned 版无 evidence 字段；原版结构 history 逐 turn 平铺 + evidence 列表，
  本适配器两种都兼容。）

原始 record 完整保留在 metadata["raw"]。
注意：文件约 265MB，json.load 峰值内存 ~2-2.5GB（16GB 机器无压力），进程退出即释放。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from schema_rsi.benchmarks.base import (
    BenchmarkCase,
    DatasetNotFoundError,
    format_case,
)
from schema_rsi.config import get_settings


def _group_history(history: list[dict]) -> list[dict]:
    """原版格式：逐 turn 的 history 按 session_id 归组（保持原顺序）。"""
    sessions: list[dict] = []
    index: dict[str, dict] = {}
    for turn in history:
        sid = str(turn.get("session_id", ""))
        if sid not in index:
            session = {"session_id": sid, "date": turn.get("timestamp"), "turns": []}
            index[sid] = session
            sessions.append(session)
        index[sid]["turns"].append(
            {
                "role": turn.get("role", "user"),
                "content": turn.get("content", ""),
                "timestamp": turn.get("timestamp"),
                "session_id": sid,
            }
        )
    return sessions


def _haystack_sessions(item: dict) -> list[dict]:
    """cleaned 版格式：haystack_* 三列表一一对应。"""
    sids = item.get("haystack_session_ids") or []
    dates = item.get("haystack_dates") or []
    sessions = []
    for i, turns in enumerate(item.get("haystack_sessions") or []):
        sid = str(sids[i]) if i < len(sids) else f"session_{i}"
        date = dates[i] if i < len(dates) else None
        sessions.append(
            {
                "session_id": sid,
                "date": date,
                "turns": [
                    {"role": t.get("role", "user"), "content": t.get("content", "")}
                    for t in turns
                ],
            }
        )
    return sessions


def _parse_item(item: dict, index: int) -> BenchmarkCase:
    history = _haystack_sessions(item)
    if not history:
        history = _group_history(item.get("history", []))
    return BenchmarkCase(
        case_id=str(item.get("question_id") or item.get("qa_id") or index),
        benchmark="longmemeval_s",
        history=history,
        question=str(item.get("question", "")),
        answer=str(item.get("answer", "")),
        category=item.get("question_type"),
        evidence=list(item["evidence"]) if item.get("evidence") else None,
        metadata={
            "raw": item,
            "question_date": item.get("question_date"),
            "answer_session_ids": list(item.get("answer_session_ids") or []),
        },
    )


class LongMemEvalDataset:
    benchmark = "longmemeval_s"

    def __init__(self, path: str | Path | None = None, data: list | None = None):
        self.path = Path(path) if path else get_settings().longmemeval_path
        self._data = data
        self._cases: list[BenchmarkCase] | None = None

    def load(self) -> None:
        if self._cases is not None:
            return
        if self._data is None:
            if not self.path.exists():
                raise DatasetNotFoundError(
                    f"LongMemEval-S 数据文件不存在: {self.path}\n"
                    "  在 config/default.yaml 的 datasets.longmemeval_path 或环境变量\n"
                    "  LONGMEMEVAL_DATASET_PATH 指向 longmemeval_s_cleaned.json（原版 S/cleaned，非 V2）。"
                )
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        if not isinstance(self._data, list):
            raise ValueError(
                f"longmemeval 数据应为 JSON list，实际为 {type(self._data).__name__}"
            )
        self._cases = [_parse_item(item, i) for i, item in enumerate(self._data)]

    @property
    def cases(self) -> list[BenchmarkCase]:
        self.load()
        return self._cases  # type: ignore[return-value]

    def count(self) -> int:
        return len(self.cases)

    def sample(self, n: int = 3) -> list[BenchmarkCase]:
        return self.cases[:n]

    def print_case(self, case_id: str | None = None) -> None:
        cases = self.cases
        case = next((c for c in cases if c.case_id == case_id), None) if case_id else cases[0]
        if case is None:
            raise KeyError(f"case not found: {case_id}")
        print(format_case(case))

    def stats(self) -> dict:
        types = Counter(c.category or "unknown" for c in self.cases)
        sessions_total = sum(c.history_stats()["sessions"] for c in self.cases)
        turns_total = sum(c.history_stats()["turns"] for c in self.cases)
        return {
            "questions": len(self.cases),
            "sessions": sessions_total,
            "turns": turns_total,
            "by_question_type": dict(types),
        }
