"""精准度优先的确定性严格判分（LoCoMo 按题型）。

与官方宽松 judge 相对的第二口径：不奖励部分正确、不允许日期容差、
不把同指代的模糊改写当全对。只对有把握的形态给出 passed/failed，
其余返回 None（交给 LLM 严格判分），避免误伤相对日期等复杂 gold。

规则：
- temporal 且 gold 解析为日期：按 gold 粒度（年/月/日/季节）精确匹配 pred 中的日期；
  gold 带相对修饰（"the week before ..."）时不判，交给 LLM
- gold 是时长（"6 months"）：数值与单位都须精确一致
- 计数/短事实题：gold 的全部实词与数字必须出现在 pred 中（数字不认词形转换）
- hedging（"probably"/"I think"/"around"）只统计不定级
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MONTHS: dict[str, int] = {}
for i, full in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1,
):
    MONTHS[full] = i
    MONTHS[full[:3]] = i
MONTHS["sept"] = 9

SEASON_MONTHS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2)}

WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(20\d{2})"
_SEASON = r"(spring|summer|fall|autumn|winter)"

_CN_DIGIT = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_num(s: str) -> int | None:
    """中文/阿拉伯数字（≤99，如 六 / 三十 / 15）转 int；解析不了返回 None。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in _CN_DIGIT:
        return _CN_DIGIT[s]
    if s == "十":
        return 10
    m = re.fullmatch(r"([一两二三四五六七八九]?)十([一两二三四五六七八九]?)", s)
    if m:
        return _CN_DIGIT.get(m.group(1), 1) * 10 + _CN_DIGIT.get(m.group(2), 0)
    return None


NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "a couple": 2, "couple": 2,
}

HEDGE_PATTERNS = [
    "not sure", "unsure", "i think", "i believe", "probably", "might", "maybe",
    "possibly", "perhaps", "likely", "appears", "seems", "around", "approximately",
    "roughly", "couldn't find", "could not find", "not mentioned", "no information",
    "don't know", "do not know", "no record", "not specified", "not clear",
    "unclear", "unable to", "no specific",
]

_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "to", "and", "or", "is", "was",
    "were", "did", "do", "does", "for", "with", "by", "from", "as", "that",
    "this", "it", "its", "his", "her", "their", "s", "he", "she", "they",
    "what", "which", "when", "where", "how", "who", "be", "been", "are",
    "have", "has", "had", "will", "would", "can", "could",
}


@dataclass
class DateRef:
    granularity: str  # "year" | "month" | "day" | "season"
    y: int | None = None
    m: int | None = None
    d: int | None = None
    season: str | None = None
    text: str = ""

    def matches(self, other: "DateRef") -> bool:
        if self.granularity == "year":
            return other.y == self.y
        if self.granularity == "month":
            return (other.y, other.m) == (self.y, self.m)
        if self.granularity == "season":
            if other.y is None or other.m is None:
                return False
            lo, hi = SEASON_MONTHS[self.season or "summer"]
            if lo <= hi:
                return other.y == self.y and lo <= other.m <= hi
            # winter 跨年：12 月属上一年
            return (other.y == self.y and other.m <= hi) or (other.y == self.y - 1 and other.m == lo)
        return (other.y, other.m, other.d) == (self.y, self.m, self.d)


def _mk(granularity: str, text: str, y=None, m=None, d=None, season=None) -> DateRef:
    return DateRef(granularity, y, m, d, season, text)


def extract_dates(text: str) -> list[DateRef]:
    """从文本抽取日期引用；无法解析出日期时返回空列表。"""
    if not text:
        return []
    t = text.lower()
    refs: list[DateRef] = []
    # ISO 2023-08-11
    for m in re.finditer(rf"\b{_YEAR}-(\d{{1,2}})-(\d{{1,2}})\b", t):
        refs.append(_mk("day", m.group(0), int(m.group(1)), int(m.group(2)), int(m.group(3))))
    # Month D, YYYY / Friday, August 11, 2023
    pat_md = rf"(?:{WEEKDAYS})?,?\s*{_MONTH}\s+{_DAY},?\s+{_YEAR}"
    for m in re.finditer(pat_md, t):
        # 分组：1=月 2=日 3=年（星期为非捕获组）
        refs.append(_mk("day", m.group(0), int(m.group(3)), MONTHS[m.group(1)], int(m.group(2))))
    # D of Month YYYY / D Month YYYY
    pat_dm = rf"\b{_DAY}\s+(?:of\s+)?{_MONTH},?\s+{_YEAR}"
    for m in re.finditer(pat_dm, t):
        # 分组：1=日 2=月 3=年
        refs.append(_mk("day", m.group(0), int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))))
    # Month YYYY（含 early/mid/late 修饰与 "January of 2023"，粒度仍为月）
    pat_my = rf"(?:early|mid|late[rs]?)?\s*{_MONTH}(?:\s+(?:of|in))?,?\s+{_YEAR}"
    for m in re.finditer(pat_my, t):
        refs.append(_mk("month", m.group(0), int(m.group(2)), MONTHS[m.group(1)]))
    # season of YYYY
    for m in re.finditer(rf"\b{_SEASON}\s+(?:of\s+|in\s+)?{_YEAR}\b", t):
        refs.append(_mk("season", m.group(0), int(m.group(2)), season=m.group(1)))
    # 裸年份（含 early/mid/late）
    for m in re.finditer(rf"\b(?:early|mid|late)\s+{_YEAR}\b", t):
        refs.append(_mk("year", m.group(0), int(m.group(1))))
    for m in re.finditer(rf"\b{_YEAR}\b", t):
        refs.append(_mk("year", m.group(0), int(m.group(1))))
    # ---- 中文日期（只加解析，判分逻辑不变）----
    # 2023年5月7日 / 2023年六月十七日
    zh_num = r"(?:\d{1,2}|[一两二三四五六七八九十]{1,3})"
    for m in re.finditer(rf"([12]\d{{3}})年\s*({zh_num})月\s*({zh_num})日", t):
        mo, d = _cn_num(m.group(2)), _cn_num(m.group(3))
        if mo and d and 1 <= mo <= 12 and 1 <= d <= 31:
            refs.append(_mk("day", m.group(0), int(m.group(1)), mo, d))
    # 2023年9月 / 2023年六月（含 初/中/底/旬 等尾饰，粒度仍为月，对齐英文 early/mid/late）
    for m in re.finditer(rf"([12]\d{{3}})年\s*({zh_num})月", t):
        mo = _cn_num(m.group(2))
        if mo and 1 <= mo <= 12:
            refs.append(_mk("month", m.group(0), int(m.group(1)), mo))
    # 2022年夏天 / 2023年冬
    for m in re.finditer(r"([12]\d{3})年\s*([春夏秋冬])(?:天|季)?", t):
        refs.append(_mk("season", m.group(0), int(m.group(1)),
                        season={"春": "spring", "夏": "summer", "秋": "fall", "冬": "winter"}[m.group(2)]))
    # 带"年"字的裸年（纯数字年已被上方英文模式覆盖）
    for m in re.finditer(r"([12]\d{3})年", t):
        refs.append(_mk("year", m.group(0), int(m.group(1))))
    return refs


# 中文相对/区间/序数表达：出现即视为相对日期，交给 LLM（对齐英文 before/ago 等的保护）
_ZH_RELATIVE = (
    "之前|以前|之后|以后|前一周|后一周|那一周|那个周|所在的|的第一|第一周|最后一周|"
    "前几天|几天前|周前|天前|月前|年前|几年|之间|上周|下周|当周|去年|明年|隔天|次日|"
    "日起|年起|日前|前的|至"
)

_RELATIVE_MARKERS = re.compile(
    r"\b(before|after|prior to|since|until|week before|week after|ago|last|next|coming|previous)\b"
    rf"|{_ZH_RELATIVE}"
)


def parse_gold_date(gold: str) -> DateRef | None:
    """temporal gold 的主日期；带相对修饰或解析不出时返回 None（交给 LLM）。"""
    refs = extract_dates(gold)
    if not refs:
        return None
    if _RELATIVE_MARKERS.search(gold.lower()):
        return None
    # gold 整体应接近一个日期短语（长度限制排除整句叙事）
    if len(gold) > 40:
        return None
    return max(refs, key=lambda r: {"day": 3, "month": 2, "season": 1, "year": 0}[r.granularity])


@dataclass
class Duration:
    value: int
    unit: str  # "day" | "week" | "month" | "year"


def parse_duration(text: str) -> Duration | None:
    t = (text or "").strip().lower().rstrip(".。")
    m = re.fullmatch(r"(?:approximately |about |around |roughly )?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(day|week|month|year)s?", t)
    if m:
        val = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1)]
        return Duration(val, m.group(2))
    # 中文时长：两周 / 三个月 / 19天 / 将近四个月（数字限 1-3 位，避免把 2023年 当时长）
    m = re.fullmatch(r"(?:将近|大约|约)?(\d{1,3}|[一两二三四五六七八九十]{1,3})\s*(天|周|星期|个月|年)", t)
    if m:
        val = _cn_num(m.group(1))
        if val is not None:
            unit = {"天": "day", "周": "week", "星期": "week", "个月": "month", "年": "year"}[m.group(2)]
            return Duration(val, unit)
    return None


def extract_durations(text: str) -> list[Duration]:
    out = []
    for m in re.finditer(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(day|week|month|year)s?\b", (text or "").lower()):
        val = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1)]
        out.append(Duration(val, m.group(2)))
    # 中文时长；(?<!\d) 排除年份（2023年）被截断误配，单字"月"不当单位（避免把日期 5月 当时长）
    for m in re.finditer(r"(?<!\d)(\d{1,3}|[一两二三四五六七八九十]{1,3})\s*(天|周|星期|个月|年)(?!\d)", (text or "").lower()):
        val = _cn_num(m.group(1))
        if val is not None:
            unit = {"天": "day", "周": "week", "星期": "week", "个月": "month", "年": "year"}[m.group(2)]
            out.append(Duration(val, unit))
    return out


def content_tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower().replace("'s", " "))
    return [t for t in toks if t not in _STOPWORDS and not t.isdigit()]


def numbers_in(text: str) -> list[int]:
    low = (text or "").lower()
    out = [int(t) for t in re.findall(r"\b\d+\b", low)]
    for w, v in NUM_WORDS.items():
        if " " not in w and re.search(rf"\b{w}\b", low):
            out.append(v)
    return out


def _token_present(tok: str, pool: set[str]) -> bool:
    if tok in pool:
        return True
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and tok[:-1] in pool:
        return True
    if tok + "s" in pool:
        return True
    return False


def strict_token_match(gold: str, pred: str) -> tuple[bool | None, str]:
    """gold 的全部实词 + 数字都出现在 pred 中才算过；gold 太长不适用返回 None。"""
    g_tokens = content_tokens(gold)
    if not g_tokens:
        return None, "gold 无实词"
    if len(g_tokens) > 12 or len(gold) > 90:
        return None, "gold 过长，交给 LLM"
    p_pool = set(content_tokens(pred))
    # 数字词（three 等）不要求字面出现，交给下方数字规则（3 == three）
    missing = [t for t in g_tokens if t not in NUM_WORDS and not _token_present(t, p_pool)]
    g_nums = set(numbers_in(gold))
    p_nums = set(numbers_in(pred))
    miss_nums = sorted(str(n) for n in g_nums - p_nums)
    if missing or miss_nums:
        return False, f"缺少: {missing + miss_nums}"
    return True, "实词与数字齐全"


def strict_check(category: str | None, gold: str, pred: str) -> dict:
    """确定性严格检查入口。返回 {passed: bool|None, rule, reason}。"""
    if category == "adversarial":
        return {"passed": None, "rule": "adversarial", "reason": "gold 为空，交给 LLM"}
    pred = pred or ""
    if category == "temporal":
        if extract_dates(gold) and _RELATIVE_MARKERS.search(gold.lower()):
            return {"passed": None, "rule": "date", "reason": "gold 含相对日期表达（before/ago 等），交给 LLM"}
        gd = parse_gold_date(gold)
        if gd is not None:
            p_refs = extract_dates(pred)
            if not p_refs:
                return {"passed": None, "rule": "date", "reason": "pred 无可解析日期，交给 LLM"}
            ok = any(gd.matches(p) for p in p_refs)
            why = f"gold={gd.text!r}({gd.granularity}) vs pred 日期 {sorted({p.text for p in p_refs})}"
            return {"passed": ok, "rule": "date", "reason": why}
        dur = parse_duration(gold)
        if dur is not None:
            p_durs = extract_durations(pred)
            same = [d for d in p_durs if d.value == dur.value and d.unit == dur.unit]
            if same:
                return {"passed": True, "rule": "duration", "reason": f"时长 {dur.value} {dur.unit} 精确一致"}
            if p_durs:
                return {"passed": False, "rule": "duration",
                        "reason": f"gold={dur.value} {dur.unit} vs pred {[f'{d.value} {d.unit}' for d in p_durs]}"}
            return {"passed": None, "rule": "duration", "reason": "pred 无时长表达，交给 LLM"}
    ok, reason = strict_token_match(gold, pred)
    return {"passed": ok, "rule": "token", "reason": reason}


def find_hedges(text: str) -> list[str]:
    t = (text or "").lower()
    return [h for h in HEDGE_PATTERNS if h in t]
