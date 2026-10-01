#!/usr/bin/env python
"""阶段 2c：独立中文图冻结配对评测（ABBA × 每臂 4 次 × 逐 call usage）。

对 2b 冻结选题包逐题：
  1. 复现当前向量检索（top45→rerank→15），与冻结 top-15 逐条 sha 比对，漂移即中止该题。
  2. BASE 臂 = 冻结 top-15；GRAPH 臂 = 向量池(45) ∪ 独立图融合候选（retrieve_fused，
     种子=base top-15）统一 rerank 取 15。逐题记录换入/挤出（含 admission_status）。
  3. 作答序列 B,G,G,B,B,G,G,B（ABBA×4），温度 0，官方 prompt（固定序号无日期，
     与 frozen-swap POC 同口径）；每答紧接 1 次官方 judge。
  4. 逐 call 记 answer/usage/judge；人工事实核侧车在裁决阶段做（先例教训：严格分不可靠）。

预算冻结（修订 2026-10-01）：≤14 题 × (8 答 + 8 判) = ≤224 LLM 调用；超 260 即停。
产出: results/analysis/m2_graph_pair_run_20261001.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PACK = PROJECT_ROOT / "results/analysis/m2_graph_eval_frozen_questions_20261001.json"
OUT = PROJECT_ROOT / "results/analysis/m2_graph_pair_run_20261001.json"
GRAPH_DB = PROJECT_ROOT / "data/graph_indep/graph.sqlite3"
ORDER = ["base", "graph", "graph", "base", "base", "graph", "graph", "base"]
CALL_CAP = 260
ANSWER_SYSTEM = "You are answering a question using retrieved memories from past conversations."
TOP_K_VECTOR = 45


def ctx_prompt(question: str, context: list[dict]) -> str:
    from schema_rsi.evaluation.prompts_official import ANSWER_GENERATION_PROMPT

    memories = "Memories in fixed retrieval order:\n" + "\n".join(
        f"{rank}. {m['content']}" for rank, m in enumerate(context, 1))
    return ANSWER_GENERATION_PROMPT.format(reference_date="2023", memories=memories, question=question)


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"拒绝重跑：{OUT} 已存在")
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    import hashlib as _h

    assert _h.sha256(json.dumps(pack["frozen_questions"], ensure_ascii=False,
                                sort_keys=True).encode()).hexdigest() == pack["pack_sha256"], "选题包被改动"

    from schema_rsi.config import get_settings
    from schema_rsi.evaluation.judge import judge_answer
    from schema_rsi.evaluation.retriever import GraphRetriever
    from schema_rsi.graph.local_store import LocalGraphStore
    from schema_rsi.llm.chat import make_chat_client
    from schema_rsi.llm.rerank import RerankClient
    from schema_rsi.memory import Mem0Backend, MemoryRecord

    from m2_build_independent_graph_zh import RateLimitedClient

    settings = get_settings(str(PROJECT_ROOT / "config/locomo_zh.yaml"))
    raw_client = make_chat_client(settings)
    raw_client._client = raw_client._client.with_options(max_retries=0, timeout=90)
    answer_client = RateLimitedClient(raw_client)
    judge_client = answer_client  # 预注册：同模型裁判（两臂公平内比），独立裁判记为后续项
    backend = Mem0Backend(settings)
    reranker = RerankClient(settings.rerank.base_url, settings.rerank.api_key, settings.rerank.model)
    store = LocalGraphStore(GRAPH_DB)
    gretr = GraphRetriever(store, settings)

    calls = 0
    report = {
        "scope": "独立中文图冻结配对评测：base=冻结当前 top-15，graph=向量池∪图融合候选统一 rerank；"
                 "ABBA×4 作答 + 同模型官方 judge；人工事实核另做。",
        "pack_sha256": pack["pack_sha256"],
        "graph_sha256": store.sha256(),
        "model": settings.llm.model,
        "order": ORDER, "call_cap": CALL_CAP,
        "limits": [
            "独立图为 S1（Entity/Event/Preference），无 S2 主题与 v8cluster；与旧混合语言运行不同构。",
            "裁判=答题同模型（glm-5.3-flash），两臂同裁判内比；独立裁判为后续项。",
            "温度 0 重复作答度量服务端噪声，非全方差；人工事实核不可省。",
        ],
        "cases": [],
    }

    for row in pack["frozen_questions"]:
        cid = row["case_id"]
        conv = cid.removeprefix("locomo_").split("_qa")[0]
        uid = f"zhfull:locomo:{conv}"
        frozen = row["current_top15_frozen"]
        frozen_map = {p["id"]: p for p in frozen}

        # 1. 复现向量检索，校验冻结漂移
        vector = backend.search_memory(row["question"], user_id=uid, top_k=TOP_K_VECTOR)
        scored = reranker.rerank(row["question"], [m.content for m in vector], top_n=15)
        live15 = [vector[i["index"]] for i in scored if 0 <= i["index"] < len(vector)][:15]
        drift = [p["id"] for p in frozen
                 if not any(l.id == p["id"] and _h.sha256(l.content.encode()).hexdigest() == p["content_sha"]
                            for l in live15)]
        if drift:
            report["cases"].append({"case_id": cid, "aborted": "retrieval_drift", "drift_ids": drift})
            print(f"  [{cid}] 检索漂移，中止该题", flush=True)
            continue

        # 2. GRAPH 臂：向量池 ∪ 图融合候选，统一 rerank
        anchors = [MemoryRecord(id=p["id"], content=p["content"],
                                metadata={"user_id": uid, "session_id": p.get("session_id")})
                   for p in frozen]
        fused = gretr.retrieve_fused(anchors, per_node_limit=15, max_candidates=30)
        pool = {m.id: {"id": m.id, "content": m.content} for m in vector}
        for f in fused:
            pool.setdefault(f["id"], {"id": f["id"], "content": f["content"],
                                      "via": f.get("via"), "support": f.get("support")})
        pool_contents = list(pool.values())
        scored2 = reranker.rerank(row["question"], [p["content"] for p in pool_contents], top_n=15)
        graph15 = [pool_contents[i["index"]] for i in scored2 if 0 <= i["index"] < len(pool_contents)][:15]
        base_ids = [p["id"] for p in frozen]
        graph_ids = [g["id"] for g in graph15]
        swapped_in = [g for g in graph15 if g["id"] not in base_ids]
        swapped_out = [frozen_map[i] for i in base_ids if i not in graph_ids]
        adm = {}
        for g in swapped_in:
            v = store.get_vertex("Memory", g["id"])
            adm[g["id"]] = (v or {}).get("properties", {}).get("admission_status")

        contexts = {"base": [dict(frozen_map[i], content=frozen_map[i]["content"]) for i in base_ids],
                    "graph": [dict(g) for g in graph15]}
        prompts = {arm: ctx_prompt(row["question"], ctx) for arm, ctx in contexts.items()}
        case = {
            "case_id": cid, "group": row["group"], "question": row["question"],
            "gold": row["gold"], "user_id": uid,
            "base_context_ids": base_ids, "graph_context_ids": graph_ids,
            "swapped_in": [{**g, "admission_status": adm.get(g["id"])} for g in swapped_in],
            "swapped_out": swapped_out,
            "must_not_evict": row.get("must_not_evict", []),
            "prompt_sha256": {a: _h.sha256(p.encode()).hexdigest() for a, p in prompts.items()},
            "calls": [],
        }
        report["cases"].append(case)
        print(f"  [{cid}] 换入 {len(swapped_in)} 换出 {len(swapped_out)}: "
              f"+{[g['id'][:8] for g in swapped_in]} -{[m['id'][:8] for m in swapped_out]}", flush=True)
        for k, arm in enumerate(ORDER, 1):
            if calls >= CALL_CAP:
                print("  !! 调用预算耗尽，停止")
                break
            try:
                answer, usage = answer_client.complete(
                    system=ANSWER_SYSTEM, user=prompts[arm], max_tokens=1200, temperature=0.0)
                calls += 1
                ok, note = judge_answer(row["question"], row["gold"], answer, judge_client,
                                        category=None, preset="official", benchmark="locomo")
                calls += 1
                case["calls"].append({"index": k, "arm": arm, "answer": answer,
                                      "usage": usage, "judge_correct": ok, "judge_note": note[:150]})
                print(f"    #{k} {arm}: judge={'✓' if ok else '✗'} {answer[:70].replace(chr(10),' ')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                case["calls"].append({"index": k, "arm": arm,
                                      "error_type": type(exc).__name__, "error": str(exc)[:200]})
                print(f"    #{k} {arm}: ERROR {str(exc)[:100]}", flush=True)
            report["calls_used"] = calls
            OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        if calls >= CALL_CAP:
            break

    report["calls_used"] = calls
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ {OUT.name} 完成，LLM 调用 {calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
