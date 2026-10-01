#!/usr/bin/env python
"""阶段 2a：独立可回滚中文图构建（SQLite 本地图 + 事实来源门 v3 准入）。

与共享 HugeGraph 实例（8080）完全隔离：
  - 存储为单个 sqlite 文件（data/graph_indep/graph.sqlite3），快照=sha256，回滚=删文件
  - 检索契约与 HugeGraphStore 相同（get_neighbors），GraphRetriever 零改动可用
  - 每条 Memory 顶点带来源门 v3 准入字段（admission_status / source_dia_ids /
    withdrawn_qualifiers），事实可追源

流程（每 conversation，门与提取均断点续跑）：
  1. 从中文 mem0 库取 zhfull:locomo:{conv} 记忆
  2. 事实来源门（v2 prompt + v3 守卫）：逐记忆对写入场次话轮判子句，
     缓存 results/analysis/m2_indep_gate_cache_20261001.jsonl
  3. StructuredExtractor S1 提取（缓存 data/graph_indep/* 播种自 graph_zh，缺口补差量）
  4. GraphBuilder.build_s1(reset=False) 写本地图；Memory 顶点回写准入字段
  5. TopicExtractor S2 + VARIANTS["v8cluster"]
  6. manifest：图 sha256、逐 label 计数、准入 tally、调用账

预算冻结（2026-10-01）：LLM 调用 ≤2,200（门 ~1,800 + 提取/主题缺口 ~140），
embedding ≤50。超出即停。共享 8080 实例零接触（本脚本不引用其配置端点）。

用法：
    .venv/bin/python -u scripts/m2_build_independent_graph_zh.py --conv conv-42   # 冒烟
    .venv/bin/python -u scripts/m2_build_independent_graph_zh.py                  # 全部
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m2_poc_fact_source_gate import SYSTEM_V2, fmt_candidate, fmt_source_turns, parse_verdict  # noqa: E402
from m2_replay_gate_guards_v3 import guard_v3  # noqa: E402

from schema_rsi.benchmarks.locomo import LocomoDataset  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.graph import GraphBuilder, make_schema_s1  # noqa: E402
from schema_rsi.graph.extractor_s1 import StructuredExtractor  # noqa: E402
from schema_rsi.graph.extractor_topics import TopicExtractor  # noqa: E402
from schema_rsi.graph.local_store import LocalGraphStore  # noqa: E402
from schema_rsi.graph.schema import make_schema_s2  # noqa: E402
from schema_rsi.graph.variants import VARIANTS  # noqa: E402
from schema_rsi.llm.chat import make_chat_client  # noqa: E402
from schema_rsi.memory import Mem0Backend, MemoryRecord  # noqa: E402

GRAPH_PATH = PROJECT_ROOT / "data/graph_indep/graph.sqlite3"
GATE_CACHE = PROJECT_ROOT / "results/analysis/m2_indep_gate_cache_20261001.jsonl"
MANIFEST = PROJECT_ROOT / "results/analysis/m2_indep_graph_manifest_20261001.json"
LLM_BUDGET = 3600  # 2026-10-01 修订：原 2,200 假设 graph_zh 缓存命中；实测 chroma 重 ingest 后
                   # 记忆 ID 再生、命中率 ~2%，提取必须重跑。补偿：砍 S2/v8cluster、只建入选题涉及 conv。
EMB_BUDGET = 100
GATE_WORKERS = 4


class Budget:
    def __init__(self) -> None:
        self.llm = 0
        self.emb = 0

    def use_llm(self, n: int = 1) -> None:
        self.llm += n
        assert self.llm <= LLM_BUDGET, f"超出 LLM 预算 {self.llm}>{LLM_BUDGET}"


class RateLimitedClient:
    """全局最小调用间隔 + 429 有界退避（30/60/120s，最多 4 次）。

    2026-10-01 全量建图在 4 并发下撞 coding 端点速率限制（error 1302），
    这是对平台限流的合规等待，非盲目重试。"""

    def __init__(self, inner, min_interval: float = 1.5):
        self._inner = inner
        self._lock = threading.Lock()
        self._min_interval = min_interval
        self._last = 0.0

    def complete(self, *args, **kwargs):
        for attempt in range(4):
            with self._lock:
                wait = self._min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                return self._inner.complete(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if "RateLimit" in type(e).__name__ or "429" in str(e)[:80]:
                    backoff = 30 * (2 ** attempt)
                    print(f"    [429] 退避 {backoff}s（第 {attempt + 1}/4 次）", flush=True)
                    time.sleep(backoff)
                    continue
                raise
        raise RuntimeError("429 退避耗尽")


def gate_one(client, budget: Budget, gi: dict, valid_refs: set[str]) -> dict:
    user = (f"源话轮（判定唯一依据）：\n{fmt_source_turns(gi)}\n\n{fmt_candidate(gi)}\n\n"
            "请输出逐子句判定 JSON。")
    verdict = None
    last_err = None
    for attempt in range(3):
        budget.use_llm()
        raw, _ = client.complete(system=SYSTEM_V2, user=user, max_tokens=2048)
        try:
            verdict = parse_verdict(raw)
            break
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:100]} | head={raw[:120]}"
            user = (f"源话轮（判定唯一依据）：\n{fmt_source_turns(gi)}\n\n{fmt_candidate(gi)}\n\n"
                    f"你上次的输出不合规（{last_err}）。重新输出严格 JSON 对象："
                    "clauses 数组 + record_action(keep|split|withdraw)，不要围栏或尾随文字。")
    if verdict is None:
        # 解析降级：单条记忆待核，不炸整体构建（记录在 guard_log）
        return {"admission_status": "待核", "source_refs": [],
                "withdrawn_qualifiers": [], "record_action": "withdraw",
                "clause_count": 0, "guard_log": [f"parse_failed: {last_err}"]}
    gated, log = guard_v3(verdict, gi)
    refs: set[str] = set()
    wq: list[str] = []
    for c in gated["clauses"]:
        refs |= {r for r in c.get("source_refs", []) if r in valid_refs}
        wq += c.get("withdrawn_qualifiers", [])
    statuses = [c["status"] for c in gated["clauses"]]
    if any(c.get("hallucinated_ref") for c in gated["clauses"]) or "withdraw" in statuses:
        status = "待核"
    elif "weaken" in statuses:
        status = "weaken"
    else:
        status = "sourced"
    return {
        "admission_status": status,
        "source_refs": sorted(refs),
        "withdrawn_qualifiers": wq,
        "record_action": gated["record_action"],
        "clause_count": len(gated["clauses"]),
        "guard_log": log,
    }


def load_gate_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if GATE_CACHE.exists():
        for line in GATE_CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["memory_id"]] = r
    return cache


def main() -> int:
    conv_filter = {
        sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--conv" and i + 1 < len(sys.argv)
    }
    settings = get_settings(str(PROJECT_ROOT / "config/locomo_zh_indep.yaml"))
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    conv_sessions: dict[str, dict[str, dict]] = {}
    conv_order: list[str] = []
    for case in ds.cases:
        cid = str(case.metadata["conversation_id"])
        if cid not in conv_sessions:
            conv_sessions[cid] = {s["session_id"]: s for s in case.history}
            conv_order.append(cid)
    convs = sorted(conv_filter) if conv_filter else conv_order
    if "--convs-from" in sys.argv:
        # 从冻结选题包取涉及 conv（阶段 2b 产物）
        pack = json.loads((PROJECT_ROOT / "results/analysis/m2_graph_eval_frozen_questions_20261001.json")
                          .read_text(encoding="utf-8"))
        convs = sorted({r["case_id"].removeprefix("locomo_").split("_qa")[0]
                        for r in pack["frozen_questions"]})
    assert set(convs) <= set(conv_sessions), "未知 conversation"

    store = LocalGraphStore(GRAPH_PATH)
    backend = Mem0Backend(settings)
    builder = GraphBuilder(store)
    raw_client = make_chat_client(settings)
    raw_client._client = raw_client._client.with_options(max_retries=0, timeout=60)
    client = RateLimitedClient(raw_client)
    extractor = StructuredExtractor(settings, client=RateLimitedClient(raw_client))
    topic_extractor = TopicExtractor(settings, client=RateLimitedClient(raw_client))
    budget = Budget()
    gate_cache = load_gate_cache()
    gate_sink = GATE_CACHE.open("a", encoding="utf-8")
    graph_cfg = settings.raw.get("graph") or {}

    def _cache_paths():
        return [settings.resolve_path(graph_cfg.get("structured_cache", "data/graph/structured_cache.json")),
                settings.resolve_path(graph_cfg.get("topic_cache", "data/graph/topic_cache.json")),
                settings.resolve_path(graph_cfg.get("memory_embeddings_cache", "data/graph/memory_embeddings.json"))]

    def _cache_sizes():
        return {Path(str(p)).name: len(json.loads(Path(str(p)).read_text(encoding="utf-8")))
                for p in _cache_paths() if Path(str(p)).exists()}

    cache_sizes_before = _cache_sizes()

    for conv in convs:
        uid = f"zhfull:locomo:{conv}"
        memories = backend.get_all_memories(user_id=uid)
        if not memories:
            print(f"[{conv}] !! 无记忆，跳过", flush=True)
            continue
        sessions = conv_sessions[conv]
        print(f"\n[{conv}] {len(memories)} 条记忆", flush=True)

        # ---- 门（并发 4，断点缓存）----
        todo = []
        for rec in memories:
            sha = hashlib.sha256(rec.content.encode()).hexdigest()
            g = gate_cache.get(rec.id)
            if not g or g.get("content_sha") != sha:
                sess = sessions.get(str((rec.metadata or {}).get("session_id") or ""))
                if not sess or not sess.get("turns"):
                    # 无可判源话轮：直接待核，不耗调用
                    gate_cache[rec.id] = {
                        "memory_id": rec.id, "content_sha": sha,
                        "admission_status": "待核", "source_refs": [],
                        "withdrawn_qualifiers": [], "record_action": "withdraw",
                        "clause_count": 0, "guard_log": ["no_session_turns"],
                    }
                    gate_sink.write(json.dumps(gate_cache[rec.id], ensure_ascii=False) + "\n")
                    gate_sink.flush()
                else:
                    todo.append((rec, sha))
        if todo:
            with ThreadPoolExecutor(max_workers=GATE_WORKERS) as ex:
                futs = {}
                for rec, sha in todo:
                    sess = sessions.get(str((rec.metadata or {}).get("session_id") or ""))
                    turns = [
                        {
                            "turn_index": j,
                            "role": t.get("role"),
                            "speaker": t.get("speaker"),
                            "content": t.get("content"),
                            "source_ref": t.get("dia_id"),
                            "image_query": t.get("image_query"),
                            "blip_caption": t.get("blip_caption"),
                        }
                        for j, t in enumerate((sess or {}).get("turns", []))
                    ] if sess else []
                    gi = {
                        "input_id": rec.id, "group": "BUILD", "benchmark": "locomo",
                        "candidate_memory": {
                            "memory_id": rec.id, "content": rec.content,
                            "written_session_id": (rec.metadata or {}).get("session_id"),
                            "written_session_date": (sess or {}).get("date"),
                            "attributed_to": (rec.metadata or {}).get("attributed_to"),
                        },
                        "source_turns": turns,
                    }
                    valid = {t["source_ref"] or f"turn_{t['turn_index']}"
                             for t in turns if t.get("content")}
                    futs[ex.submit(gate_one, client, budget, gi, valid)] = (rec, sha)
                for fu in futs:
                    rec, sha = futs[fu]
                    res = fu.result()
                    gate_cache[rec.id] = {"memory_id": rec.id, "content_sha": sha, **res}
                    gate_sink.write(json.dumps(gate_cache[rec.id], ensure_ascii=False) + "\n")
                    gate_sink.flush()
            print(f"  门: 新判 {len(todo)} 条（累计 LLM {budget.llm}）", flush=True)
        missing_gate = [r.id for r in memories if r.id not in gate_cache]
        assert not missing_gate, f"门缓存缺失: {missing_gate[:3]}"

        # ---- S1 提取（播种缓存，缺口补差量）----
        t0 = time.time()
        extraction = extractor.extract(memories)
        n_e = sum(len(v.get("entities", [])) for v in extraction.values())
        n_v = sum(len(v.get("events", [])) for v in extraction.values())
        n_p = sum(len(v.get("preferences", [])) for v in extraction.values())
        print(f"  S1 提取: {n_e}实体/{n_v}事件/{n_p}偏好 ({time.time()-t0:.0f}s)", flush=True)

        # ---- 建图 ----
        report = builder.build_s1(memories, extraction, make_schema_s1(), reset=False)
        print(f"  S1 写入: {report.label_counts} edges={report.edges_written}", flush=True)

        # ---- 准入字段回写 ----
        for rec in memories:
            g = gate_cache[rec.id]
            store.upsert_vertex("Memory", rec.id, {
                "admission_status": g["admission_status"],
                "source_dia_ids": ",".join(g["source_refs"][:20]),
                "withdrawn_qualifiers": ";".join(g["withdrawn_qualifiers"][:5]),
            })

        # ---- S2 主题 + v8cluster（--s1-only 跳过；修订后默认 s1-only）----
        if "--s1-only" not in sys.argv:
            t0 = time.time()
            topics = topic_extractor.extract(memories)
            builder.build_concepts(memories, topics)
            counts = VARIANTS["v8cluster"].build(store, memories, settings)
            print(f"  S2+v8cluster: {counts} ({time.time()-t0:.0f}s)", flush=True)

    gate_sink.close()

    # ---- manifest ----
    cache_sizes_after = _cache_sizes()
    cache_delta = {k: cache_sizes_after.get(k, 0) - cache_sizes_before.get(k, 0)
                   for k in cache_sizes_after}
    labels = {vl.name: store.count_vertices(vl.name)
              for vl in list(make_schema_s1().vertex_labels) + list(make_schema_s2().vertex_labels)}
    tally: dict[str, int] = {}
    for g in gate_cache.values():
        tally[g["admission_status"]] = tally.get(g["admission_status"], 0) + 1
    manifest = {
        "scope": "独立中文图构建（SQLite 本地图+来源门 v3 准入），与共享 8080 实例零接触。",
        "config": "config/locomo_zh_indep.yaml",
        "graph_path": str(GRAPH_PATH.relative_to(PROJECT_ROOT)),
        "graph_sha256": store.sha256(),
        "vertex_counts": labels,
        "edge_counts": {e: store.count_edges(e) for e in (
            "MENTIONS", "DESCRIBES_EVENT", "EXPRESSES_PREFERENCE", "RELATED_TO",
            "PARTICIPATES_IN", "HAS_PREFERENCE", "HAS_TOPIC", "V8_IN_CLUSTER")},
        "gate_tally": tally,
        "gate_cache_size": len(gate_cache),
        "budget": {"llm_calls_gate": budget.llm, "llm_cap": LLM_BUDGET,
                   "cache_delta": cache_delta,
                   "note": "cache_delta 即提取/主题/嵌入的补差调用量上限（含失败重试为 0 的约定）"},
        "convs": convs,
        "shared_instance_untouched": "本脚本未引用 8080 端点；基线顶点数见 m2_indep_build_baseline_8080.json",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n✓ manifest →", MANIFEST.relative_to(PROJECT_ROOT))
    print(json.dumps({k: manifest[k] for k in ("vertex_counts", "gate_tally", "budget")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
