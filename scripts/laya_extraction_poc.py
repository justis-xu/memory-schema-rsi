#!/usr/bin/env python
"""Laya 提取质量门控 POC：验证"P(B) 风险分"能否区分有漏提取/干净的 session。

背景：mem0 提取存在已知漏提取（results/attribution_split_A.json 的"提取丢失"桶，
关键词归因+人工校准，精度约七成）。想法是每次按 session 提取后让 Laya 只答三个
二分类 choice 问题（missing/unsupported/conflict），取 P(B) 当风险分，超阈值才
升级 glm-5.3 二审。本脚本对单个对话（默认 conv-42，29 个 session）：

  1. 逐 session 构造 state（[原文对话] 全部 turns + [当前提取结果] 该 session 的
     flash 记忆），按 src/schema_rsi/llm/laya.py 的协议 POST /v1/decisions；
  2. 对比两个服务：laya 完整版 vs decider-2b；
  3. 以真值（漏提取 evidence dia_id 落在的 session 为正例）算 AUC
     （Mann-Whitney U 手写）、正/负例均值、阈值表 0.3~0.7；
  4. payoff 参照：nf53 重灌库对这些漏提取条目的覆盖数（_covered，复用
     scripts/verify_extraction_tier.py）。

只读数据：数据集 / 主向量库 / 真值文件一律只读；唯一写入是输出 JSON（默认
results/laya_poc/conv42_poc.json，可用 --out 改）。端点优先本机
6006(laya)/6008(decider)，不通走公网（域名以 laya.py:28 的 seetacloud 为准；
ask 中的 seattlecloud 拼写在公网 DNS 为 NXDOMAIN，已核实为笔误）。公网若被
本机 fake-IP DNS 劫持（198.18.0.0/15，TLS 握手被切），用 DoH 解析真实 IP
后 patch getaddrinfo 直连。

中文轨道（locomo-zh）：--config config/locomo_zh.yaml --prefix zhfull
--audit results/laya_poc_zh/audit.json [--out results/laya_poc_zh/conv42_poc.json]。
数据集/向量库按 config 加载；提取记忆取 {prefix}:locomo:{conv}（注意拼写是
locomo）按 metadata.session_id 分组；真值标签改读 audit.json（正例=该 session
missing_facts 非空，由 scripts/zh_session_audit.py 用 glm-5.3 造）；nf53 payoff
跳过（中文轨道无非 flash 对照库，输出里注明）。三问 choice 判据保持现有中文
原文，AUC/阈值表/双服务对比不变。

用法：
  .venv/bin/python scripts/laya_extraction_poc.py --conv conv-42 [--limit 2]          # 英文轨道
  .venv/bin/python scripts/laya_extraction_poc.py --config config/locomo_zh.yaml \
      --prefix zhfull --audit results/laya_poc_zh/audit.json --limit 2                # 中文轨道
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 复用 _covered/_content_words

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 关 mem0 PostHog 噪声，保 stdout 干净

import requests  # noqa: E402

from verify_extraction_tier import _covered, _content_words  # noqa: E402
from zh_session_audit import ground_keyphrase  # noqa: E402  # 审计同款关键短语锚定

# ---- 端点 -----------------------------------------------------------------
# 公网域名与 src/schema_rsi/llm/laya.py:28 DEFAULT_LAYA_BASE 一致（seetacloud）；
# decider 是独立实例，前缀 uu（ask 原文的 seattlecloud 为笔误，公网 NXDOMAIN）。
LOCAL_ENDPOINTS = {
    "laya": "http://127.0.0.1:6006",
    "decider": "http://127.0.0.1:6008",
}
PUBLIC_ENDPOINTS = {
    "laya": "https://u1172328-a3uc-8f103d7d.weste.seetacloud.com:8443",
    "decider": "https://uu1172328-a3uc-8f103d7d.weste.seetacloud.com:8443",
}

# ---- 三个 choice 问题（判据原文，不让服务输出原因） --------------------------
QUESTIONS = {
    "missing": {
        "type": "choice",
        "instructions": "当前提取结果是否遗漏了原文中的有效信息？",
        "criteria": {
            "A": "当前结果完整，没有遗漏需要提取的信息",
            "B": "存在至少一项应该提取但未被提取的信息",
        },
    },
    "unsupported": {
        "type": "choice",
        "instructions": "当前提取结果是否缺乏原文证据支持？",
        "criteria": {
            "A": "全部条目都有原文支持",
            "B": "存在至少一项原文不支持",
        },
    },
    "conflict": {
        "type": "choice",
        "instructions": "原文是否存在与当前提取结果冲突、否定或更新的信息？",
        "criteria": {
            "A": "无冲突",
            "B": "存在冲突或更新",
        },
    },
}

_DIA_RE = re.compile(r"D(\d+):(\d+)")
_LENGTH_ERROR_RE = re.compile(
    r"too long|length|context (window|length|size)|maximum.*chars?|token limit|exceed", re.I
)
_STATE_LIMIT = 12000  # 长度兜底：实测双服务接受 88k+，此值仅为保险
_SEM = threading.Semaphore(2)  # 红线：并发 ≤2（比 laya.py 的 4 更紧）


# ---- 网络：fake-IP DNS 绕过 -------------------------------------------------
def _doh_resolve(host: str) -> str | None:
    """通过 Cloudflare DoH 解析真实 A 记录（本机 DNS 可能被 fake-IP 代理劫持）。"""
    try:
        r = requests.get(
            f"https://1.1.1.1/dns-query?name={host}&type=A",
            headers={"accept": "application/dns-json"},
            timeout=10,
        )
        ans = r.json().get("Answer") or []
        ips = [a["data"] for a in ans if a.get("type") == 1]
        return ips[0] if ips else None
    except Exception:
        return None


_DNS_BYPASS: dict[str, str] = {}
_orig_getaddrinfo = socket.getaddrinfo


def _patched_getaddrinfo(host, *args, **kwargs):
    if isinstance(host, str) and host in _DNS_BYPASS:
        return _orig_getaddrinfo(_DNS_BYPASS[host], *args, **kwargs)
    return _orig_getaddrinfo(host, *args, **kwargs)


socket.getaddrinfo = _patched_getaddrinfo


def _is_fake_ip(host: str) -> bool:
    try:
        infos = _orig_getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip in ipaddress.ip_network("198.18.0.0/15"):
            return True
    return False


_PROBE_QS = {"probe": {"type": "choice", "instructions": "Is this a probe?",
                       "criteria": {"A": "yes", "B": "no"}}}


def pick_endpoint(service: str) -> tuple[str, str]:
    """本机优先，不通走公网；公网被 fake-IP DNS 劫持时 DoH 直连。返回 (url, mode)。"""
    probe = {"state": {"body": "Connectivity probe."}, "questions": _PROBE_QS}
    for url, mode in ((LOCAL_ENDPOINTS[service], "local"), (PUBLIC_ENDPOINTS[service], "public")):
        try:
            r = requests.post(f"{url}/v1/decisions", json=probe, timeout=8)
            if r.status_code == 200:
                return url, mode
        except requests.RequestException:
            pass
    host = PUBLIC_ENDPOINTS[service].split("//", 1)[1].rsplit(":", 1)[0]
    ip = _doh_resolve(host)
    if ip and _is_fake_ip(host):
        _DNS_BYPASS[host] = ip
        try:
            r = requests.post(f"{PUBLIC_ENDPOINTS[service]}/v1/decisions", json=probe, timeout=15)
            if r.status_code == 200:
                return PUBLIC_ENDPOINTS[service], "public+doh"
        except requests.RequestException:
            pass
    raise SystemExit(f"[FATAL] {service} 端点全部不可达：local={LOCAL_ENDPOINTS[service]} "
                     f"public={PUBLIC_ENDPOINTS[service]}（DoH 解析 {host} -> {ip}）")


# ---- 决策调用（协议严格对齐 laya.py:96-116） --------------------------------
class LengthError(Exception):
    pass


def decide(base_url: str, state_body: str, questions: dict, retries: int = 2) -> dict:
    """POST /v1/decisions：body={"state":{"body":...},"questions":...}，timeout=15，
    422/500 不重试同参数，其余失败退避重试 2 次。长度/上下文错误抛 LengthError。"""
    last = ""
    for attempt in range(retries + 1):
        try:
            with _SEM:
                r = requests.post(
                    f"{base_url}/v1/decisions",
                    json={"state": {"body": state_body}, "questions": questions},
                    timeout=15,
                )
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if _LENGTH_ERROR_RE.search(last) or r.status_code in (413,):
                raise LengthError(last)
            if r.status_code in (422, 500):  # 格式错/引擎异常：不重试同参数
                break
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException as e:
            last = str(e)[:200]
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"decide failed: {last}")


def parse_pb(resp: dict, question_key: str) -> tuple[float, str | None]:
    """answers.<key>.probabilities["B"] → P(B)。解析不到视为协议失配，抛错。"""
    ans = (resp.get("answers") or {}).get(question_key)
    if not isinstance(ans, dict):
        raise ValueError(f"answers.{question_key} 缺失: {json.dumps(resp, ensure_ascii=False)[:200]}")
    probs = ans.get("probabilities")
    if not isinstance(probs, dict) or not isinstance(probs.get("B"), (int, float)):
        raise ValueError(f"answers.{question_key}.probabilities.B 缺失: "
                         f"{json.dumps(ans, ensure_ascii=False)[:200]}")
    return float(probs["B"]), ans.get("choice")


# ---- state 构造与长度兜底 ---------------------------------------------------
def build_state(session: dict, memory_texts: list[str]) -> str:
    turns = "\n".join(f"{t['speaker']}: {t['content']}" for t in session["turns"])
    mems = "\n".join(f"- {m}" for m in memory_texts) if memory_texts else "（该 session 无提取记忆）"
    return f"[原文对话]\n{turns}\n\n[当前提取结果]\n{mems}"


def truncate_middle(state: str, limit: int = _STATE_LIMIT) -> str:
    """长度兜底：截掉原文中段，保留头尾（POC 实测双服务接受 88k+，此路径未触发过）。"""
    if len(state) <= limit:
        return state
    keep = (limit - 60) // 2
    cut = len(state) - 2 * keep
    return state[:keep] + f"\n…[中段截断 {cut} 字符]…\n" + state[-keep:]


def score_session(base_url: str, state: str) -> tuple[dict, dict, bool, int, dict]:
    """单 session 单服务：返回 ({q: P(B)}, {q: choice}, truncated, state_chars, raw)。"""
    truncated = False
    try:
        resp = decide(base_url, state, QUESTIONS)
    except LengthError:
        state = truncate_middle(state)
        truncated = True
        resp = decide(base_url, state, QUESTIONS)
    pb, choices = {}, {}
    for q in QUESTIONS:
        pb[q], choices[q] = parse_pb(resp, q)
    return pb, choices, truncated, len(state), resp


# ---- 真值 / 数据装配 ---------------------------------------------------------
def load_truth(conv: str, audit_path: str | None = None) -> tuple[dict[str, list[str]], list[dict]]:
    """返回 (session_id -> miss_items 文本列表, 逐条 miss item dict)。

    audit 模式（中文轨道）：正例 = audit.json 中该 session missing_facts 非空
    （强模型审计 + 关键短语锚定修正，见 scripts/zh_session_audit.py），truth_items 为空。

    英文轨道（默认）：attribution_split_A.json 中 bucket=='提取丢失' 且 conv 匹配
    的条目，用其 case_id 回数据集查 evidence dia_id（D<k>:<t>），k 即漏提取发生
    的 session 号。"""
    if audit_path:
        p = Path(audit_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        data = json.loads(p.read_text(encoding="utf-8"))
        session_miss = {
            str(s["session_id"]): list(s.get("missing_facts") or [])
            for s in data.get("sessions", [])
            if s.get("missing_facts")  # 正例 = missing_facts 非空
        }
        return session_miss, []

    from schema_rsi.benchmarks.locomo import LocomoDataset

    split = json.loads((PROJECT_ROOT / "results/attribution_split_A.json").read_text(encoding="utf-8"))
    miss = [it for it in split if it.get("bucket") == "提取丢失" and it.get("conv") == conv]

    ds = LocomoDataset()
    ds.load()
    ev_by_case = {c.case_id: (c.evidence or []) for c in ds.cases
                  if c.metadata.get("conversation_id") == conv}
    session_miss: dict[str, list[str]] = {}
    for it in miss:
        ks = set()
        for dia in ev_by_case.get(it["case_id"]) or []:
            m = _DIA_RE.fullmatch(str(dia))
            if m:
                ks.add(int(m.group(1)))
        for k in ks:
            session_miss.setdefault(f"session_{k}", []).append(it["text"])
    return session_miss, miss


def load_memories(conv: str, config_path: str | None = None, prefix: str = "full",
                  nf_prefix: str | None = None) -> tuple[dict[str, list[str]], list[str]]:
    """按 config 的后端取 {prefix}:locomo:{conv} 按 session_id 分组（只读）。

    对照库（第二个返回值）：显式传 --nf-prefix 时读 {nf_prefix}:locomo:{conv}
    （中文轨道非 flash 重灌库）；否则仅英文轨道默认读 nf53:locomo:{conv}。"""
    from schema_rsi.config import get_settings
    from schema_rsi.memory import Mem0Backend

    backend = Mem0Backend(get_settings(config_path))
    by_sid: dict[str, list[str]] = {}
    for m in backend.get_all_memories(user_id=f"{prefix}:locomo:{conv}"):
        by_sid.setdefault(str((m.metadata or {}).get("session_id")), []).append(m.content)
    if nf_prefix:
        nf = [m.content for m in backend.get_all_memories(user_id=f"{nf_prefix}:locomo:{conv}")]
    elif config_path is None and prefix == "full":  # 英文轨道默认 nf53 对照
        nf = [m.content for m in backend.get_all_memories(user_id=f"nf53:locomo:{conv}")]
    else:
        nf = []
    return by_sid, nf


def load_sessions(conv: str, config_path: str | None = None) -> list[dict]:
    from schema_rsi.benchmarks.locomo import LocomoDataset

    if config_path:  # 中文轨道：数据集路径来自 config（locomo10_zh.json）
        from schema_rsi.config import get_settings

        ds = LocomoDataset(get_settings(config_path).locomo_path)
    else:
        ds = LocomoDataset()  # 默认英文配置（config/default.yaml 的 locomo10.json）
    ds.load()
    case = next((c for c in ds.cases if c.metadata.get("conversation_id") == conv), None)
    if case is None:
        raise SystemExit(f"[FATAL] 对话 {conv} 不在数据集中")
    return case.history


def calc_nf_payoff(session_miss: dict[str, list[str]], sessions_by_sid: dict[str, dict],
                   nf_mems: list[str], nf_prefix: str) -> dict:
    """中文闭环 payoff：audit.json 里每条有效 missing_fact，取审计脚本同款
    『连续 ≥4 字且最长』关键短语（对该 session 原文锚定，ground_keyphrase），
    在 {nf_prefix}:locomo:{conv} 库的全部记忆里做子串包含匹配
    （中文没有空格分词，一律子串包含，不做词边界匹配）。"""
    detail: list[dict] = []
    covered = 0
    for sid, facts in session_miss.items():
        sess = sessions_by_sid.get(sid)
        if sess is None:
            for fact in facts:
                detail.append({"session_id": sid, "fact": fact, "keyphrase": None,
                               "covered": False, "note": "session 不在数据集"})
            continue
        text = "\n".join(f"{t['speaker']}: {t['content']}" for t in sess["turns"])
        for fact in facts:
            key = ground_keyphrase(fact, text)
            hit = key is not None and any(key in m for m in nf_mems)
            if hit:
                covered += 1
            detail.append({"session_id": sid, "fact": fact, "keyphrase": key, "covered": hit})
    return {"漏点条数": len(detail), "nf_covered": covered, "nf_prefix": nf_prefix, "明细": detail}


# ---- 指标 -------------------------------------------------------------------
def auc_mann_whitney(pos: list[float], neg: list[float]) -> float | None:
    """手写 Mann-Whitney U：AUC = P(pos 分 > neg 分) + 0.5*P(相等)。不依赖 scipy。"""
    if not pos or not neg:
        return None
    u = 0.0
    for p in pos:
        for n in neg:
            u += 1.0 if p > n else (0.5 if p == n else 0.0)
    return u / (len(pos) * len(neg))


def service_metrics(sessions: list[dict], service: str) -> dict:
    ok = [s for s in sessions if s.get(service) and not s[service].get("error")]
    risk = [s[service]["missing"] for s in ok]  # 主风险分 = missing P(B)（漏提取门控信号）
    pos = [r for s, r in zip(ok, risk) if s["has_known_miss"]]
    neg = [r for s, r in zip(ok, risk) if not s["has_known_miss"]]
    out = {
        "n_scored": len(ok),
        "risk": "missing P(B)",
        "auc": auc_mann_whitney(pos, neg),
        "mean_risk_pos": sum(pos) / len(pos) if pos else None,
        "mean_risk_neg": sum(neg) / len(neg) if neg else None,
        "auc_by_question": {
            q: auc_mann_whitney(
                [s[service][q] for s in ok if s["has_known_miss"]],
                [s[service][q] for s in ok if not s["has_known_miss"]],
            )
            for q in QUESTIONS
        },
        "threshold_table": {
            f"{thr:.1f}": {
                "升级数": sum(1 for r in risk if r >= thr),
                "其中有真漏数": sum(1 for s, r in zip(ok, risk) if r >= thr and s["has_known_miss"]),
                "漏检数": sum(1 for s, r in zip(ok, risk) if r < thr and s["has_known_miss"]),
            }
            for thr in (0.3, 0.4, 0.5, 0.6, 0.7)
        },
    }
    return out


# ---- 主流程 -----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conv", default="conv-42")
    ap.add_argument("--limit", type=int, default=None, help="只打前 N 个 session（冒烟用）")
    ap.add_argument("--print-raw", action="store_true",
                    help="打印每个请求的服务原始 JSON 响应（冒烟核验解析用）")
    ap.add_argument("--config", default=None,
                    help="配置轨道（中文：config/locomo_zh.yaml；缺省英文轨道 default.yaml）")
    ap.add_argument("--prefix", default="full", help="提取记忆的 user_id 前缀（中文轨道 zhfull）")
    ap.add_argument("--audit", default=None,
                    help="真值 audit.json（中文轨道强模型审计，正例=missing_facts 非空）；"
                         "缺省用英文轨道 attribution_split_A.json")
    ap.add_argument("--nf-prefix", dest="nf_prefix", default=None,
                    help="非 flash 对照库前缀（中文轨道如 zhnf53，需先用 scripts/ingest_conv.py "
                         "--model glm-5.3 灌库）；给定时对 audit 有效漏点算 nf payoff")
    ap.add_argument("--out", default=None,
                    help="输出 JSON 路径；缺省英文 results/laya_poc/conv42_poc.json、"
                         "中文轨道（--config/--audit 时）results/laya_poc_zh/conv42_poc.json")
    args = ap.parse_args()

    print(f"[1/4] 选择端点（本机优先，不通走公网）…")
    endpoints = {}
    for service in ("laya", "decider"):
        url, mode = pick_endpoint(service)
        endpoints[service] = {"url": url, "mode": mode}
        print(f"  {service:8} -> {url}  ({mode})")

    track = {"config": args.config, "prefix": args.prefix, "audit": args.audit,
             "nf_prefix": args.nf_prefix}
    print(f"[2/4] 装配只读数据（conv={args.conv}, track={track}）…")
    # 全量 sessions 供 payoff 锚定（不受 --limit 影响）；打分只用前 N 个
    all_sessions = load_sessions(args.conv, args.config)
    sessions_by_sid = {s["session_id"]: s for s in all_sessions}
    sessions_raw = all_sessions[: args.limit] if args.limit else all_sessions
    session_miss, truth_items = load_truth(args.conv, args.audit)
    by_sid, nf_mems = load_memories(args.conv, args.config, args.prefix, args.nf_prefix)
    print(f"  sessions={len(sessions_raw)}  有漏提取标记={sum(1 for s in sessions_raw if s['session_id'] in session_miss)}"
          + (f"  nf对照库({args.nf_prefix})={len(nf_mems)} 条" if args.nf_prefix else ""))

    print("[3/4] 逐 session 调两个决策服务（并发≤2，顺序执行）…")
    results = []
    for i, sess in enumerate(sessions_raw, 1):
        sid = sess["session_id"]
        state = build_state(sess, by_sid.get(sid, []))
        row = {
            "session_id": sid,
            "n_turns": len(sess["turns"]),
            "n_memories": len(by_sid.get(sid, [])),
            "has_known_miss": sid in session_miss,
            "miss_items": session_miss.get(sid, []),
            "state_chars": len(state),
        }
        for service in ("laya", "decider"):
            try:
                pb, choices, truncated, n_chars, raw = score_session(endpoints[service]["url"], state)
                row[service] = {**pb, "choices": choices, "truncated": truncated,
                                "state_chars": n_chars}
                if args.print_raw:
                    print(f"  ---- {sid} {service} RAW RESPONSE ----")
                    print("  " + json.dumps(raw, ensure_ascii=False))
                print(f"  [{i:>2}/{len(sessions_raw)}] {sid:11} miss={int(row['has_known_miss'])} "
                      f"{service:8} P(B) missing={pb['missing']:.4f} unsupported={pb['unsupported']:.4f} "
                      f"conflict={pb['conflict']:.4f}" + ("  [截断]" if truncated else ""))
            except Exception as e:  # 单请求失败不中断整体
                row[service] = {"error": str(e)[:300]}
                print(f"  [{i:>2}/{len(sessions_raw)}] {sid:11} {service:8} ERROR: {str(e)[:120]}")
        results.append(row)

    print("[4/4] 汇总指标 …")
    if args.nf_prefix and args.audit:
        # 中文闭环 payoff：升级 glm-5.3 二审能救回多少（nf 库对 audit 有效漏点的覆盖）
        payoff = calc_nf_payoff(session_miss, sessions_by_sid, nf_mems, args.nf_prefix)
        print(f"  payoff: 漏点 {payoff['漏点条数']} 条，{args.nf_prefix} 库覆盖 "
              f"{payoff['nf_covered']} 条（子串包含，无词边界）")
    elif args.audit:
        payoff = {"skipped": True, "note": "未给 --nf-prefix，中文轨道 nf payoff 跳过"}
    elif args.nf_prefix:
        payoff = {"skipped": True, "note": "--nf-prefix 仅在 --audit 模式下生效（有效漏点来自 audit.json）"}
    else:
        # payoff：全部 conv 内漏提取条目（不受 --limit 影响，按真值清单聚合）
        payoff = {
            "漏提取条数": len(truth_items),
            "nf53_覆盖数": sum(
                1 for it in truth_items
                if _covered(it.get("words") or _content_words(it.get("text", "")), nf_mems)
            ),
        }
    output = {
        "conv": args.conv,
        "track": track,
        "endpoints_used": endpoints,
        "sessions": results,
        "metrics": {service: service_metrics(results, service) for service in ("laya", "decider")},
        "payoff": payoff,
    }
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    elif args.config or args.audit:  # 中文轨道缺省写到 zh 目录
        out_path = PROJECT_ROOT / "results/laya_poc_zh/conv42_poc.json"
    else:
        out_path = PROJECT_ROOT / "results/laya_poc/conv42_poc.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {
        "conv": args.conv,
        "track": track,
        "n_sessions": len(results),
        "n_pos": sum(1 for r in results if r["has_known_miss"]),
        "n_neg": sum(1 for r in results if not r["has_known_miss"]),
        "endpoints_used": endpoints,
        "metrics": output["metrics"],
        "payoff": {k: v for k, v in payoff.items() if k != "明细"},  # 明细只在输出文件里
        "output_file": str(out_path.relative_to(PROJECT_ROOT)),
    }
    print("\n=== POC METRICS SUMMARY (JSON) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("=== END SUMMARY ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
