#!/usr/bin/env python
"""提取层模型档位验证：定位"提取丢失"点，并在重灌库上复查覆盖（零判题成本）。

背景：终审结论把剩余 33pp 缺口的最大头归为提取层丢失（single-hop 76.7% 封顶）。
本脚本不改任何现有数据，做两件事：

  find   对终审精准判分里非 exact 的题，检查答案缺失事实（judge.missing /
         gold 关键词）是否存在于 flash 主记忆库；输出"库里没有"的条目清单
         JSON，供重灌后复查。纯本地查询，不调 LLM。
  check  把同一清单在重灌库（--prefix 前缀隔离的 user）上复查，输出
         flash❌→非flash 的逐条对比。

用法：
  .venv/bin/python scripts/verify_extraction_tier.py find \
      --final results/full_locomo_finalA_base_20260925_013909.jsonl \
      --regrade results/precision_regrade_A.jsonl \
      --out results/extraction_miss_A.json
  .venv/bin/python scripts/verify_extraction_tier.py check \
      --items results/extraction_miss_A.json --prefix nf53
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "was", "are", "were",
        "for", "on", "at", "with", "by", "it", "s", "not", "before", "after",
        "correct", "specific", "missing", "answer", "question", "day", "week",
        "month", "year", "time", "date", "period", "point",
        # 判分抱怨类元词（"hedged precision"/"definitive without hedging" 等），
        # 不构成缺失事实，过滤掉避免把口径问题算成提取丢失
        "hedged", "hedging", "hedge", "definitive", "definitively", "precision",
        "precise", "without", "rather", "than", "gold", "extra", "unverified",
        "likely", "incorrect", "mislabeled", "mixing", "mixed", "adds", "adding",
        "specifies", "specifying", "single", "entities", "entity", "contains",
        "containing", "instead", "needs", "need", "should", "must", "vague",
        "unclear", "correctly", "exactly", "around", "roughly", "early", "late",
        "when", "while", "since", "been", "being", "have", "has", "had", "did",
        "does", "done", "there", "their", "them", "then", "this", "that", "those",
        "these", "from", "will", "would", "could", "might", "may", "also", "into",
        "his", "her", "she", "him", "they", "its", "one", "about", "attribution",
        # 第二轮人工核验补充的判分元词（count/hedges/qualified 类抱怨措辞）
        "count", "hedges", "hedging", "least", "qualified", "possibly", "commitment",
        "level", "offers", "offering", "unnamed", "verifiable", "unsupported",
        "alternative", "equal", "clarinet", "granularity", "commits", "commit"}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def _content_words(s: str) -> list[str]:
    return [w for w in _norm(s).split() if w not in STOP and len(w) > 2]


def _evidence_text(raw_entry: dict, evidence: list) -> tuple[str, list[int]]:
    """收集 evidence dia_id（D<k>:<t>）对应的原始 turn 文本 + 涉及的 session 号。

    raw_entry 是 dataset 顶层条目，session 列表在其 "conversation" 子 dict 里。
    """
    conv = raw_entry.get("conversation") or raw_entry
    texts, sessions = [], []
    for dia in evidence or []:
        m = re.fullmatch(r"D(\d+):(\d+)", str(dia))
        if not m:
            continue
        k = int(m.group(1))
        sessions.append(k)
        for turn in conv.get(f"session_{k}", []):
            if str(turn.get("dia_id", "")) == str(dia):
                texts.append(str(turn.get("text", "")))
    return " ".join(texts), sorted(set(sessions))


def _covered(item_words: list[str], texts: list[str]) -> bool:
    """条目 ≥ 一半内容词在同一段文本里共现才算命中（词边界，防子串假阳性）。"""
    if not item_words:
        return False
    need = max(1, (len(item_words) + 1) // 2)
    for t in texts:
        toks = set(_norm(t).split())
        if sum(1 for w in item_words if w in toks) >= need:
            return True
    return False


def _store_covered(words: list[str], mems: list[tuple[str, str | None]],
                   ev_sessions: list[int]) -> bool:
    """库内存在性：证据 session 的记忆 ≥ 半数词命中即可；其他 session 的记忆
    要求全词命中（防止无关话题记忆撞词造成召回/提取误判）。mems=(content, session_id)。"""
    ev = {f"session_{k}" for k in ev_sessions}
    half = max(1, (len(words) + 1) // 2)
    for content, sid in mems:
        toks = set(_norm(content).split())
        n = sum(1 for w in words if w in toks)
        if sid in ev and n >= half:
            return True
        if n == len(words):
            return True
    return False


def _load_store(user_ids: set[str]) -> dict[str, list[str]]:
    from schema_rsi.config import get_settings
    from schema_rsi.memory import Mem0Backend

    backend = Mem0Backend(get_settings())
    store = {}
    for uid in sorted(user_ids):
        store[uid] = [m.content for m in backend.get_all_memories(user_id=uid)]
    return store


def _load_store_meta(user_ids: set[str]) -> dict[str, list[tuple[str, str | None]]]:
    """同 _load_store，但保留 (content, session_id) 供证据 session 约束判断。"""
    from schema_rsi.config import get_settings
    from schema_rsi.memory import Mem0Backend

    backend = Mem0Backend(get_settings())
    store: dict[str, list[tuple[str, str | None]]] = {}
    for uid in sorted(user_ids):
        store[uid] = [
            (m.content, (m.metadata or {}).get("session_id")) for m in backend.get_all_memories(user_id=uid)
        ]
    return store


def cmd_find(args) -> int:
    from schema_rsi.benchmarks.locomo import LocomoDataset

    final = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.final)}
    regrade = [json.loads(l) for l in open(args.regrade)]

    ds = LocomoDataset()
    ds.load()
    raw_by_conv, evidence_by_case = {}, {}
    for c in ds.cases:
        raw_by_conv[str(c.metadata.get("conversation_id"))] = c.metadata["raw"]
        evidence_by_case[c.case_id] = c.evidence

    store = _load_store({r["metadata"]["user_id"] for r in final.values()
                         if r["metadata"].get("user_id")})

    items, stats = [], Counter()
    per_conv = defaultdict(lambda: Counter())
    for rg in regrade:
        cat = rg.get("category")
        if cat == "adversarial" or rg.get("final") == "exact":
            continue
        case_id = rg["case_id"]
        row = final.get(case_id)
        if not row:
            continue
        uid = row["metadata"]["user_id"]
        conv = case_id.split("_qa")[0].replace("locomo_", "")
        memories = store.get(uid, [])

        # 缺失事实清单：优先 judge.missing（partial 为主）；wrong 无 missing 用 gold
        raw_missing = list((rg.get("judge") or {}).get("missing") or [])
        targets = [(t, "judge_missing") for t in raw_missing if _content_words(t)]
        if not targets:
            targets = [(rg.get("gold", ""), "gold")]
        ev_text, ev_sessions = _evidence_text(raw_by_conv.get(conv, {}),
                                              evidence_by_case.get(case_id) or [])

        for text, src in targets:
            words = _content_words(text)
            if not words:  # 纯判分抱怨（hedging/口径），不是缺失事实
                stats["meta_only"] += 1
                continue
            # 事实词必须严格出现在证据原文里（≥ 半数共现），否则不算提取的责任
            in_evidence = bool(ev_text) and _covered(words, [ev_text])
            if not in_evidence:
                stats["not_in_evidence"] += 1
                continue
            in_store = _covered(words, memories)
            tag = "in_store" if in_store else "not_in_store"
            stats[tag] += 1
            per_conv[conv][tag] += 1
            if tag == "not_in_store":
                items.append({
                    "case_id": case_id, "conv": conv, "user_id": uid,
                    "category": cat, "src": src, "item": text, "words": words,
                    "in_evidence": in_evidence, "evidence_sessions": ev_sessions,
                    "question": rg.get("question", row["question"])[:120],
                    "gold": rg.get("gold", ""),
                    "grade": rg.get("final"),
                })

    n_case = len({r["case_id"] for r in regrade
                  if r.get("category") != "adversarial" and r.get("final") != "exact"})
    print(f"非 exact 非 adversarial 题：{n_case}，缺失事实条目归因：")
    for k in ("not_in_store", "in_store", "not_in_evidence", "meta_only"):
        print(f"  {k:16} {stats[k]:4}  ({stats[k] / max(1, sum(stats.values())):.0%})")
    print("\n分 conversation（not_in_store = flash 库里没有该事实）：")
    for conv in sorted(per_conv, key=lambda c: -per_conv[c]["not_in_store"]):
        c = per_conv[conv]
        if c["not_in_store"]:
            print(f"  {conv:10} not_in_store={c['not_in_store']:3}  "
                  f"in_store={c['in_store']:3}  not_in_evidence={c['not_in_evidence']:3}")

    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"\n待复查条目（not_in_store）{len(items)} 条 → {args.out}")
    return 0


def cmd_check(args) -> int:
    items = json.loads(Path(args.items).read_text())
    store = _load_store({f"{args.prefix}:locomo:{it['conv']}" for it in items})
    flash = _load_store({it["user_id"] for it in items})

    fixed, still = [], []
    for it in items:
        uid_nf = f"{args.prefix}:locomo:{it['conv']}"
        it["nf_covered"] = _covered(it["words"], store.get(uid_nf, []))
        (fixed if it["nf_covered"] else still).append(it)

    by_conv = defaultdict(lambda: [0, 0])
    for it in items:
        by_conv[it["conv"]][0 if it["nf_covered"] else 1] += 1
    print(f"复查 {len(items)} 条 flash 库缺失事实（前缀 {args.prefix} 的重灌库）：")
    for conv in sorted(by_conv):
        ok, miss = by_conv[conv]
        print(f"  {conv:10} 补上 {ok:3} / 仍缺 {miss:3}")
    print(f"\n合计：补上 {len(fixed)} / {len(items)}"
          f"（{len(fixed) / max(1, len(items)):.0%}），仍缺 {len(still)}")

    print("\n[补上样例]（最多 8 条）")
    for it in fixed[:8]:
        print(f"  {it['case_id']} [{it['category']}/{it['grade']}] {it['item'][:70]}")
    print("\n[仍缺样例]（最多 8 条）")
    for it in still[:8]:
        print(f"  {it['case_id']} [{it['category']}/{it['grade']}] {it['item'][:70]}")

    # 记忆条数对比（同对话 flash vs 非 flash）
    print("\n[每对话记忆条数] flash vs 重灌")
    for conv in sorted({it["conv"] for it in items}):
        n_flash = len(flash.get(f"full:locomo:{conv}", []))
        n_nf = len(store.get(f"{args.prefix}:locomo:{conv}", []))
        print(f"  {conv:10} flash={n_flash:4}  nf={n_nf:4}  Δ{n_nf - n_flash:+d}")
    Path(args.items).with_suffix(".checked.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1))
    return 0


def cmd_split(args) -> int:
    """四级硬拆：每条缺失事实按管线顺序归入第一个该它负责的环节。

    提取丢失（对话明说、库里没有）→ 召回丢失（库里有、没进作答上下文）
    → 上下文已有（进了上下文仍没答出 → 作答/判分）→ 对话没明说（提取无责，
    日期计算/综合推理）。互斥：命中前一级就不再往下算。
    """
    import sys as _sys
    from collections import Counter, defaultdict

    from schema_rsi.benchmarks.locomo import LocomoDataset

    final = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.final)}
    regrade = [json.loads(l) for l in open(args.regrade)]

    ds = LocomoDataset()
    ds.load()
    raw_by_conv, ev_by_case, full, windows = {}, {}, {}, {}
    for c in ds.cases:
        conv = str(c.metadata.get("conversation_id"))
        raw_by_conv[conv] = c.metadata["raw"]
        ev_by_case[c.case_id] = c.evidence
        if conv not in full:
            turns = [str(t.get("content", "")) for s in c.history for t in s.get("turns", [])]
            full[conv] = " ".join(turns)
            windows[conv] = [" ".join(turns[i:i + 2]) for i in range(len(turns))]

    def in_conv(words, conv, ev_text):
        if ev_text and _covered(words, [ev_text]):
            return True
        if len(words) <= 4 and _covered(words, [full.get(conv, "")]):
            return True
        return _covered(words, windows.get(conv, []))

    store = _load_store_meta({r["metadata"]["user_id"] for r in final.values()
                              if r["metadata"].get("user_id")})

    BUCKETS = ["提取丢失", "召回丢失", "上下文已有", "对话没明说"]
    items = []
    for rg in regrade:
        if rg.get("category") == "adversarial" or rg.get("final") == "exact":
            continue
        case_id = rg["case_id"]
        row = final.get(case_id)
        if not row:
            continue
        uid = row["metadata"]["user_id"]
        conv = case_id.split("_qa")[0].replace("locomo_", "")
        memories = store.get(uid, [])
        ctx = [m.get("content", "") for m in row.get("retrieved_memories", [])]
        ev_text, ev_sessions = _evidence_text(raw_by_conv.get(conv, {}),
                                              ev_by_case.get(case_id) or [])

        raw_missing = list((rg.get("judge") or {}).get("missing") or [])
        targets = [(t, "judge_missing") for t in raw_missing if _content_words(t)]
        if not targets:
            targets = [(rg.get("gold", ""), "gold")]

        for text, src in targets:
            words = _content_words(text)
            if not words:
                continue
            if not in_conv(words, conv, ev_text):
                tag = "对话没明说"
            elif not _store_covered(words, memories, ev_sessions):
                tag = "提取丢失"
            elif not _covered(words, ctx):
                tag = "召回丢失"
            else:
                tag = "上下文已有"
            items.append({"case_id": case_id, "conv": conv, "user_id": uid,
                          "category": rg.get("category"), "grade": rg.get("final"),
                          "src": src, "text": text, "words": words, "bucket": tag})

    n_q = len({it["case_id"] for it in items})
    print(f"缺失事实条目 {len(items)}（{n_q} 道非 exact 非 adversarial 题）四级硬拆：\n")
    print(f"{'桶':6} {'条目':>4} {'条目%':>6} {'涉及题目':>4} {'题目%':>6}")
    for b in BUCKETS:
        rows = [it for it in items if it["bucket"] == b]
        qs = {it["case_id"] for it in rows}
        print(f"{b:8} {len(rows):4} {len(rows)/len(items):6.0%} {len(qs):4} "
              f"{len(qs)/n_q:6.0%}")
    # 题目级互斥归因（管线顺序，前级失败优先）
    by_q = defaultdict(set)
    for it in items:
        by_q[it["case_id"]].add(it["bucket"])
    prim = Counter()
    for q, bs in by_q.items():
        prim[next((b for b in BUCKETS if b in bs), "上下文已有")] += 1
    print(f"\n题目级主归因（一题只算最先失败的环节）：{dict(prim)}")
    # 分题型
    print("\n分题型（条目）：")
    for cat in ("single-hop", "multi-hop", "temporal", "open-domain"):
        c = Counter(it["bucket"] for it in items if it["category"] == cat)
        n = sum(c.values())
        print(f"  {cat:12} n={n:3}  " +
              "  ".join(f"{b}={c.get(b,0)}" for b in BUCKETS))

    for b in BUCKETS:
        rows = [it for it in items if it["bucket"] == b][:5]
        print(f"\n[{b} 样例]")
        for it in rows:
            print(f"  {it['case_id'].replace('locomo_','')} [{it['category']}] {it['text'][:75]}")
    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"\n→ {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("--final", required=True)
    f.add_argument("--regrade", required=True)
    f.add_argument("--out", default="results/extraction_miss_A.json")
    c = sub.add_parser("check")
    c.add_argument("--items", required=True)
    c.add_argument("--prefix", default="nf53")
    s = sub.add_parser("split")
    s.add_argument("--final", required=True)
    s.add_argument("--regrade", required=True)
    s.add_argument("--out", default="results/attribution_split_A.json")
    args = p.parse_args()
    return (cmd_find if args.cmd == "find"
            else cmd_check if args.cmd == "check" else cmd_split)(args)


if __name__ == "__main__":
    sys.exit(main())
