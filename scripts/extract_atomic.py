#!/usr/bin/env python
"""原子事实全量提取 + 写入独立向量池 schema_rsi_atomic。

对 LoCoMo 10 个对话的全部会话跑原子提取（缓存续跑），把事实嵌入
（MaaS qwen3.7，与主池同空间）写入 data/vector_store/chroma 的
schema_rsi_atomic 集合。主池不动，历史基线臂不受影响。

用法：.venv/bin/python scripts/extract_atomic.py [--conv conv-26]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    from schema_rsi.benchmarks.locomo import LocomoDataset
    from schema_rsi.config import get_settings
    from schema_rsi.extraction.atomic import AtomicExtractor, ATOMIC_SYSTEM  # noqa: F401
    from schema_rsi.memory.chroma_direct import ChromaDirectBackend  # 复用其嵌入客户端

    settings = get_settings()
    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    conv_filter = {sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--conv" and i + 1 < len(sys.argv)}

    extractor = AtomicExtractor(settings)
    emb_backend = ChromaDirectBackend(settings)  # 只用它的 _embed
    # 目标集合：主 chroma 里的 schema_rsi_atomic
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.resolve_path(settings.raw["mem0"]["vector_store_path"])))
    col = client.get_or_create_collection("schema_rsi_atomic")

    by_conv: dict[str, list] = {}
    for case in ds.cases:
        conv = str(case.metadata.get("conversation_id"))
        if conv_filter and conv not in conv_filter:
            continue
        by_conv.setdefault(conv, []).append(case)

    t0 = time.time()
    total = 0
    for conv, cases in by_conv.items():
        uid = f"full:locomo:{conv}"
        case0 = cases[0]
        speakers = f"{case0.metadata.get('speaker_a', 'A')} and {case0.metadata.get('speaker_b', 'B')}"
        # 按 session 分组 turns（case.history 已是 session 列表）
        facts_meta = []
        for sess in case0.history:
            sid = str(sess.get("session_id") or f"{conv}:{len(facts_meta)}")
            sdate = str(sess.get("date") or "")
            turns = sess.get("turns") or []
            if not turns:
                continue
            facts = extractor.extract_session(f"{uid}:{sid}", turns, sdate, speakers)
            for f in facts:
                facts_meta.append({
                    "id": extractor.fact_id(uid, sid, f),
                    "fact": f,
                    "meta": {"user_id": uid, "session_id": sid, "session_date": sdate,
                             "source": "atomic", "conversation": conv},
                })
        existing = set()
        if col.count():
            existing = {i for i in col.get()["ids"]} if col.get()["ids"] else set()
        fresh = [x for x in facts_meta if x["id"] not in existing]
        if fresh:
            embs = emb_backend._embed([x["fact"] for x in fresh])  # noqa: SLF001
            for i in range(0, len(fresh), 200):
                batch = fresh[i : i + 200]
                col.add(ids=[x["id"] for x in batch],
                        documents=[x["fact"] for x in batch],
                        embeddings=embs[i : i + 200],
                        metadatas=[x["meta"] for x in batch])
        print(f"  {conv}: {len(facts_meta)} 原子事实（新增 {len(fresh)}）")
        total += len(facts_meta)
    print(f"合计 {total} 条原子事实，集合总量 {col.count()}（{time.time()-t0:.0f}s）")
    print(f"缓存：{len(extractor._cache)} 会话")  # noqa: SLF001
    return 0


if __name__ == "__main__":
    sys.exit(main())
