#!/usr/bin/env python
"""从旧 mem0/chroma 库导出全部记忆，用 BigModel embedding 重建直连向量库。

背景：百炼密钥失效，向量层切 BigModel。旧库不动；新库 data/chroma_bm/
（id 不变，保证图顶点主键仍然对得上）。嵌入走 BigModel（settings.embedding
已指向 BigModel 的前提下——本脚本会临时覆盖 embedding 配置）。

用法：.venv/bin/python scripts/rebuild_vector_bm.py [--model embedding-3]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BM_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"


def main() -> int:
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "embedding-3"
    from schema_rsi.config import get_settings
    from schema_rsi.memory import Mem0Backend

    settings = get_settings()
    # 旧库读取（get_all 不需要嵌入调用；embedder 指向百炼也无妨）
    old = Mem0Backend(settings)
    users = sorted({str(c.metadata.get("conversation_id")) for c in _locomo_convs(settings)})
    records = []
    for conv in users:
        uid = f"full:locomo:{conv}"
        rs = old.get_all_memories(user_id=uid)
        records.extend(rs)
        print(f"  {conv}: {len(rs)} 条")
    print(f"旧库导出 {len(records)} 条记忆")

    # 新库写入：embedding 配置临时指向 BigModel
    settings.raw.setdefault("embedding", {})
    import schema_rsi.config as cfg

    settings.embedding.base_url = BM_BASE
    settings.embedding.api_key = settings.llm.api_key
    settings.embedding.model = model
    from schema_rsi.memory.chroma_direct import ChromaDirectBackend

    new = ChromaDirectBackend(settings)
    if new._col.count():  # noqa: SLF001
        print(f"新库已有 {new._col.count()} 条，跳过重建（要重来先删 data/chroma_bm）")
        return 0

    t0 = time.time()
    embs = new._embed([r.content for r in records])  # noqa: SLF001
    for i in range(0, len(records), 200):
        batch = records[i : i + 200]
        new._col.add(  # noqa: SLF001
            ids=[r.id for r in batch],
            documents=[r.content for r in batch],
            embeddings=embs[i : i + 200],
            metadatas=[dict(r.metadata or {}) for r in batch],
        )
    print(f"新库写入 {new._col.count()} 条（{model}，{time.time()-t0:.0f}s）")
    # 冒烟：检索一次
    hits = new.search_memory("What car does Evan drive?", user_id="full:locomo:conv-49", top_k=3)
    for h in hits:
        print("  检索命中:", h.content[:70])
    return 0


def _locomo_convs(settings):
    from schema_rsi.benchmarks.locomo import LocomoDataset

    ds = LocomoDataset(settings.locomo_path)
    ds.load()
    return ds.cases


if __name__ == "__main__":
    sys.exit(main())
