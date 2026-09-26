"""precision.py 确定性严格判分的单元测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_rsi.evaluation.precision import (
    extract_dates,
    find_hedges,
    parse_duration,
    parse_gold_date,
    strict_check,
    strict_token_match,
)


def _grans(text):
    return [(r.granularity, r.y, r.m, r.d) for r in extract_dates(text)]


def test_extract_dates_formats():
    assert ("day", 2023, 1, 15) in _grans("January 15, 2023")
    assert ("day", 2023, 1, 15) in _grans("15 January 2023")
    assert ("day", 2023, 8, 11) in _grans("Friday, August 11, 2023")
    assert ("day", 2023, 8, 11) in _grans("on 2023-08-11")
    assert ("month", 2023, 8, None) in _grans("August 2023")
    assert ("month", 2023, 8, None) in _grans("in mid-August, 2023")
    assert ("season", 2022, None, None) in _grans("approximately summer of 2022")
    assert ("year", 2023, None, None) in _grans("early 2023")
    assert ("year", 2022, None, None) in _grans("during 2022")


def test_parse_gold_date_relative_defers():
    assert parse_gold_date("The week before 7 July 2023") is None
    assert parse_gold_date("A few years ago") is None
    gd = parse_gold_date("January 2023")
    assert gd is not None and gd.granularity == "month"


def test_strict_check_temporal_date():
    ok = strict_check("temporal", "January 2023", "It was in January of 2023.")
    assert ok["passed"] is True
    miss = strict_check("temporal", "January 2023", "Around the end of June 2023.")
    assert miss["passed"] is False
    coarse = strict_check("temporal", "January 15, 2023", "In January 2023")
    assert coarse["passed"] is False  # 粒度不足：对月无日
    year_only = strict_check("temporal", "2022", "Around mid-2022")
    assert year_only["passed"] is True
    defer = strict_check("temporal", "The Friday before 14 August 2023", "Friday, August 11, 2023")
    assert defer["passed"] is None  # 相对 gold 交给 LLM


def test_strict_check_duration():
    assert strict_check("temporal", "6 months", "They stayed there for six months.")["passed"] is True
    assert strict_check("temporal", "6 months", "About five months.")["passed"] is False
    assert strict_check("temporal", "two years", "A long period")["passed"] is None


def test_strict_token_match():
    ok, _ = strict_token_match("DoorDash", "He worked at DoorDash in 2023.")
    assert ok is True
    miss, why = strict_token_match("Sennheiser headphones, Logitech mouse, gaming desk",
                                   "John bought Sennheiser headphones and a Logitech mouse.")
    assert miss is False and "desk" in why
    num, _ = strict_token_match("three", "Joanna has written four screenplays.")
    assert num is False
    num_word, _ = strict_token_match("three", "The total is 3.")
    assert num_word is True
    none, _ = strict_token_match("By reminding herself of her successes and progress, having a support system, and focusing on the big goal and the reasons she started the business, which keep her motivated despite challenges",
                                 "She focuses on her goal.")
    assert none is None


def test_strict_check_adversarial_defers():
    assert strict_check("adversarial", "", "Some fabricated answer.")["passed"] is None


def test_find_hedges():
    h = find_hedges("I think it was DoorDash, probably in early 2023.")
    assert "i think" in h and "probably" in h


def test_parse_duration():
    d = parse_duration("approximately 6 months")
    assert d and d.value == 6 and d.unit == "month"
    assert parse_duration("winning first place") is None


def test_extract_dates_zh():
    assert ("day", 2023, 5, 7) in _grans("2023年5月7日")
    assert ("day", 2023, 6, 17) in _grans("2023年六月十七日")
    assert ("month", 2023, 6, None) in _grans("2023年六月")
    assert ("month", 2023, 8, None) in _grans("2023年8月初")
    assert ("season", 2022, None, None) in _grans("2022年夏天")
    assert ("year", 2013, None, None) in _grans("2013年")
    # 无年月日不解析（对齐英文：无年日期不产生引用，交给 LLM）
    assert _grans("8月13日") == []
    assert _grans("八月") == []


def test_parse_gold_date_zh_relative_defers():
    assert parse_gold_date("2023年5月25日之前的那个周日") is None
    assert parse_gold_date("2023年8月11日的两周前") is None
    assert parse_gold_date("4月3日至9日那一周") is None
    assert parse_gold_date("从2016年起") is None
    assert parse_gold_date("2023年11月22日的前几天") is None
    gd = parse_gold_date("2023年5月7日")
    assert gd is not None and gd.granularity == "day"


def test_strict_check_zh_temporal():
    ok = strict_check("temporal", "2023年5月7日", "卡罗琳于2023年5月7日参加了LGBTQ互助小组。")
    assert ok["passed"] is True and ok["rule"] == "date"
    miss = strict_check("temporal", "2023年5月7日", "卡罗琳于2023年6月7日参加了活动。")
    assert miss["passed"] is False
    coarse = strict_check("temporal", "2023年5月7日", "那是2023年5月的事。")
    assert coarse["passed"] is False  # 粒度不足：对日无日
    season = strict_check("temporal", "2022年夏天", "她在2022年6月去了。")
    assert season["passed"] is True
    defer = strict_check("temporal", "2023年7月6日之前的那一周", "2023年7月1日那周。")
    assert defer["passed"] is None  # 相对 gold 交给 LLM


def test_duration_zh():
    d = parse_duration("两周")
    assert d and d.value == 2 and d.unit == "week"
    d = parse_duration("将近四个月")
    assert d and d.value == 4 and d.unit == "month"
    assert parse_duration("三个月") is not None
    assert parse_duration("2022年") is None  # 年份不是时长
    assert parse_duration("10年前") is None  # 相对表达
    assert strict_check("temporal", "六个月", "他们认识大约六个月了。")["passed"] is True
    # 年份不应被截断成时长（2023年 -> 23年）
    from schema_rsi.evaluation.precision import extract_durations
    assert all(not (d.unit == "year" and d.value > 12) for d in extract_durations("从2023年5月到2024年，共花了3个月。"))
    assert any(d.value == 3 and d.unit == "month" for d in extract_durations("共花了3个月"))
