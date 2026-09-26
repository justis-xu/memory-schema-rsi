"""统一 BenchmarkCase 与 DatasetAdapter 协议。

数据流：
    Dataset 原始结构 → Dataset Adapter（locomo.py / longmemeval.py）→ BenchmarkCase → 统一评测管线

原则：不为统一而丢字段 —— 原始 record 完整保留在 metadata["raw"]。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class BenchmarkCase:
    case_id: str
    benchmark: str
    history: list  # [{"session_id": str|None, "date": str|None, "turns": [{"role","content",...}]}]
    question: str
    answer: str
    category: str | None = None
    evidence: list | None = None
    metadata: dict = field(default_factory=dict)  # 原始记录存 metadata["raw"]

    def history_turns(self, max_turns: int | None = None) -> list[dict]:
        """摊平全部 turns（可选截断，用于控制 ingest 的 LLM 调用量）。"""
        turns: list[dict] = []
        for session in self.history:
            for t in session.get("turns", []):
                turns.append(t)
                if max_turns is not None and len(turns) >= max_turns:
                    return turns
        return turns

    def history_stats(self) -> dict:
        return {
            "sessions": len(self.history),
            "turns": sum(len(s.get("turns", [])) for s in self.history),
        }


@runtime_checkable
class DatasetAdapter(Protocol):
    benchmark: str

    def load(self) -> None: ...
    def count(self) -> int: ...
    def sample(self, n: int = 3) -> list[BenchmarkCase]: ...
    def print_case(self, case_id: str | None = None) -> None: ...


class DatasetNotFoundError(RuntimeError):
    pass


def _norm_text(text: str) -> str:
    import re

    text = str(text).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def answer_in_window(case: BenchmarkCase, max_turns: int | None = None) -> bool:
    """粗筛：标准答案（归一化后）是否作为子串出现在前 max_turns 轮对话文本里。

    用于构造"gold 事实在 ingest 窗口内"的评测子集（LongMemEval 这类没有
    evidence 标注的数据集的通用判据；答案需要推理/日期换算的 case 会被漏掉，
    所以是"粗筛"——宁缺勿滥，保证基线可区分）。
    """
    if max_turns is None:
        max_turns = 10**9
    haystack = _norm_text(" ".join(t.get("content", "") for t in case.history_turns(max_turns)))
    return bool(_norm_text(case.answer)) and _norm_text(case.answer) in haystack


def parse_session_date(date_str) -> str | None:
    """把数据集的 session 日期解析为 ISO 字符串（解析失败返回 None）。

    支持的格式（实测）：
      LoCoMo:        "4:04 pm on 20 January, 2023"
      LongMemEval:   "2023/05/20 (Sat) 02:21"
    """
    if not date_str:
        return None
    from datetime import datetime

    s = str(date_str).strip()
    formats = (
        "%I:%M %p on %d %B, %Y",       # LoCoMo
        "%I:%M %p on %d %B %Y",
        "%Y/%m/%d (%a) %H:%M",          # LongMemEval
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    return None


def _truncate(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_case(case: BenchmarkCase, turns_per_session: int = 3) -> str:
    """打印一条完整 case（长历史按 session 截断展示）。"""
    lines = [
        "=" * 72,
        f"case_id   : {case.case_id}",
        f"benchmark : {case.benchmark}",
        f"category  : {case.category}",
        f"question  : {case.question}",
        f"answer    : {case.answer}",
        f"evidence  : {case.evidence if case.evidence else '(none)'}",
        f"history   : {case.history_stats()}",
    ]
    for session in case.history:
        sid = session.get("session_id")
        date = session.get("date")
        lines.append(f"  --- session {sid} {('(' + date + ')') if date else ''} ---")
        for t in session.get("turns", [])[:turns_per_session]:
            lines.append(f"    [{t.get('role', '?')}] {_truncate(t.get('content', ''))}")
        n = len(session.get("turns", []))
        if n > turns_per_session:
            lines.append(f"    ... ({n - turns_per_session} more turns)")
    lines.append("=" * 72)
    return "\n".join(lines)
