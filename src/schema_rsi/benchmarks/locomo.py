"""LoCoMo 数据集适配器。

官方 locomo10.json 真实结构（已实测核对）：
顶层是 10 个 conversation 的 list，每个 entry：
  - "sample_id": "conv-26"（conversation 标识）
  - "conversation": {
        "speaker_a"/"speaker_b": 说话人名,
        "session_1" .. "session_N": turn 列表（部分 session 只有 date 没有 turn 列表）,
        "session_N_date_time": session 日期,
    }
    turn: {"dia_id": "D1:3"(编码 session), "speaker", "text", 可选多模态字段}
  - "qa": [{"question", "answer", "category"(int 1-5), "evidence": [dia_id, ...]}]
  - "event_summary" / "observation" / "session_summary": 附加材料（保留在 raw 里）

不修改任何数据内容；原始 record 全量保留在 BenchmarkCase.metadata["raw"]。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from schema_rsi.benchmarks.base import (
    BenchmarkCase,
    DatasetNotFoundError,
    format_case,
)
from schema_rsi.config import get_settings

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")

# LoCoMo 官方 QA 类别编号 → 名称（对照 mem0 官方 benchmark 代码核对：
# 计数分布 282/321/96/841/446 与本数据完全吻合；注意官方 J-score 排除 adversarial）
CATEGORY_MAP = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# 官方 J-score 只对类别 1-4 计分（adversarial 排除）
CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]


def _turn_text(turn: dict) -> str:
    text = turn.get("text", "") or ""
    blip = turn.get("blip_caption") or turn.get("query")
    if blip:
        text += f" [shared image: {turn.get('query', '')} | {turn.get('blip_caption', '')}]"
    return text.strip()


def _image_urls(turn: dict) -> list[str]:
    raw = turn.get("img_url")
    if isinstance(raw, list):
        return [str(url) for url in raw if url]
    return [str(raw)] if raw else []


def _parse_conversation(entry: dict, conv_index: int) -> list[BenchmarkCase]:
    conv = entry.get("conversation") or entry  # 兼容未来平铺结构
    conv_id = str(entry.get("sample_id") or entry.get("conversation_id") or conv_index)
    speaker_a = conv.get("speaker_a", "A")
    speaker_b = conv.get("speaker_b", "B")

    session_keys = []
    for key in conv:
        m = _SESSION_KEY_RE.match(key)
        if m and isinstance(conv[key], list) and conv[key]:
            session_keys.append((int(m.group(1)), key))
    session_keys.sort()

    sessions: list[dict] = []
    for num, key in session_keys:
        date = conv.get(f"session_{num}_date_time")
        turns = []
        for turn in conv[key]:
            speaker = turn.get("speaker", "")
            turns.append(
                {
                    # Mem0 侧统一 user/assistant 角色；原始 speaker 保留在 turn 里
                    "role": "user" if speaker == speaker_a else "assistant",
                    "content": _turn_text(turn),
                    "speaker": speaker,
                    "dia_id": turn.get("dia_id", ""),
                    "image_urls": _image_urls(turn),
                    "blip_caption": turn.get("blip_caption"),
                    "image_query": turn.get("query"),
                }
            )
        sessions.append({"session_id": f"session_{num}", "date": date, "turns": turns})

    qa_list = entry.get("qa") or entry.get("qa_pairs") or []
    cases = []
    for i, qa in enumerate(qa_list):
        raw_category = qa.get("category")
        category = CATEGORY_MAP.get(raw_category, str(raw_category) if raw_category else None)
        cases.append(
            BenchmarkCase(
                case_id=f"locomo_{conv_id}_qa{i}",
                benchmark="locomo",
                history=sessions,
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                category=category,
                evidence=list(qa.get("evidence") or []) or None,
                metadata={
                    "raw": entry,  # 原始 conversation record（同 conversation 的 case 共享引用）
                    "qa_raw": qa,
                    "conversation_id": conv_id,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                },
            )
        )
    return cases


def evidence_in_window(case: BenchmarkCase, max_turns: int) -> bool:
    """LoCoMo 精筛：case.evidence 里的 dia_id（"D<k>:<t>"，k=session 号，t=轮号）
    对应的全部证据 turn 是否都落在前 max_turns 轮内。

    （LoCoMo 的答案常需日期换算，答案原文不一定出现在对话里，所以用
    evidence 定位而不是 answer 子串匹配。）
    """
    if not case.evidence:
        return False
    # session_id -> 该 session 之前的累计 turn 数
    offsets: dict[int, int] = {}
    offset = 0
    for session in case.history:
        sid = str(session.get("session_id", ""))
        m = re.fullmatch(r"session_(\d+)", sid)
        if m:
            offsets[int(m.group(1))] = offset
        offset += len(session.get("turns", []))
    for dia_id in case.evidence:
        m = re.fullmatch(r"D(\d+):(\d+)", str(dia_id))
        if not m:
            return False
        sess, turn = int(m.group(1)), int(m.group(2))
        base = offsets.get(sess)
        if base is None:  # 证据 session 不在 history（被 date-only 跳过的 session）
            return False
        if base + turn > max_turns:
            return False
    return True


class LocomoDataset:
    benchmark = "locomo"

    def __init__(self, path: str | Path | None = None, data: list | None = None):
        self.path = Path(path) if path else get_settings().locomo_path
        self._data = data
        self._cases: list[BenchmarkCase] | None = None

    # ---- loading ----

    def load(self) -> None:
        if self._cases is not None:
            return
        if self._data is None:
            if not self.path.exists():
                raise DatasetNotFoundError(
                    f"LoCoMo 数据文件不存在: {self.path}\n"
                    "  运行 ./scripts/download_data.sh 下载官方英文版（2.8MB），\n"
                    "  或在 config/default.yaml / 环境变量 LOCOMO_DATASET_PATH 指向你自己的文件。"
                )
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        if not isinstance(self._data, list):
            raise ValueError(f"locomo 数据应为 JSON list，实际为 {type(self._data).__name__}")

        cases: list[BenchmarkCase] = []
        for idx, entry in enumerate(self._data):
            cases.extend(_parse_conversation(entry, idx))
        self._cases = cases

    # ---- DatasetAdapter 协议 ----

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

    # ---- 数据集统计 ----

    def stats(self) -> dict:
        conversations = {}
        for c in self.cases:
            conv_id = c.metadata.get("conversation_id")
            conversations.setdefault(conv_id, c)
        sessions_total = sum(len(c.history) for c in conversations.values())
        turns_total = sum(c.history_stats()["turns"] for c in conversations.values())
        categories = Counter(c.category or "unknown" for c in self.cases)
        return {
            "conversations": len(conversations),
            "sessions": sessions_total,
            "turns": turns_total,
            "qa_total": len(self.cases),
            "qa_by_category": dict(categories),
        }
