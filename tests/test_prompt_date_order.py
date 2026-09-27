"""Date headers in answer prompts must match the stated chronological order."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_rsi.evaluation.prompts_official import format_memories_official
from schema_rsi.evaluation.prompts_lme_official import format_memories_lme
from schema_rsi.memory.base import MemoryRecord


def test_locomo_sorts_by_parsed_date_not_leading_clock_text():
    records = [
        MemoryRecord("late", "late fact", {"session_date": "1:08 pm on 11 August, 2023"}),
        MemoryRecord("early", "early fact", {"session_date": "7:48 pm on 21 May, 2023"}),
        MemoryRecord("middle", "middle fact", {"session_date": "10:52 am on 27 July, 2023"}),
        MemoryRecord("unknown", "unknown fact", {}),
    ]
    prompt = format_memories_official(records)

    assert prompt.index("early fact") < prompt.index("middle fact") < prompt.index("late fact")
    assert prompt.index("late fact") < prompt.index("unknown fact")
    assert "(unknown date) unknown fact" in prompt


def test_longmemeval_year_first_dates_remain_newest_first():
    records = [
        MemoryRecord("old", "old fact", {"session_date": "2023/05/20 (Sat) 02:21"}),
        MemoryRecord("new", "new fact", {"session_date": "2023/06/02 (Fri) 15:30"}),
    ]
    prompt = format_memories_lme(records)

    assert prompt.index("new fact") < prompt.index("old fact")
