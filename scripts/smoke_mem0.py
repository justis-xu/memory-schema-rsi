#!/usr/bin/env python
"""Mem0 冒烟测试：5 轮对话 → add（user_id 隔离）→ search 断言 → get_all → reset。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.memory import Mem0Backend  # noqa: E402


CONVERSATION = [
    {"role": "user", "content": "我叫李雷，最近搬到了杭州，在一家做智能手表的公司当产品经理。"},
    {"role": "assistant", "content": "好的李雷，已了解你在杭州做智能手表产品经理。"},
    {"role": "user", "content": "我养了一只叫 Milo 的英短猫，它三岁了，最爱吃冻干鸡肉。"},
    {"role": "assistant", "content": "Milo 听起来很可爱，我会记住它。"},
    {"role": "user", "content": "对了，我下周三下午要去看牙医，在文一路口腔诊所。"},
    {"role": "assistant", "content": "已记下你下周三的牙医预约。"},
]

QUERIES = [
    ("用户的猫叫什么名字？", ["milo"]),
    ("用户住在哪个城市？", ["杭州", "hangzhou"]),  # mem0 提取的记忆可能是英文，双语断言
]


def main() -> int:
    settings = get_settings()
    if not (settings.llm.api_key and settings.embedding.api_key):
        print("✗ Mem0 smoke 需要可用的 LLM + Embedding 端点：")
        print("  cp .env.example .env 并填写 LLM_API_KEY / EMBEDDING_API_KEY 等字段后重跑。")
        return 2

    print(f"llm       = {settings.llm.model} @ {settings.llm.base_url}")
    print(f"embedding = {settings.embedding.model} @ {settings.embedding.base_url}")
    print(f"vector    = chroma @ {settings.mem0.get('vector_store_path')}")

    backend = Mem0Backend(settings)
    user_id = "smoke_user_1"

    print(f"\n[0] reset（保证冒烟可重复执行）")
    backend.reset_memories()
    print("    已清空")

    print(f"\n[1] add: {len(CONVERSATION)} 条消息（user_id={user_id}）")
    records = backend.add_memory(user_id, CONVERSATION, metadata={"benchmark": "smoke", "case_id": "smoke_1"})
    print(f"    产生/更新记忆 {len(records)} 条:")
    for r in records[:8]:
        print(f"      - [{r.id}] {r.content[:60]}")

    print("\n[2] search:")
    ok = len(records) > 0
    for query, keywords in QUERIES:
        hits = backend.search_memory(query, user_id=user_id, top_k=3)
        contents = [h.content for h in hits]
        hit = any(any(k.lower() in c.lower() for k in keywords) for c in contents)
        ok = ok and hit
        print(f"    Q: {query}")
        for h in hits[:2]:
            print(f"      ({h.score if h.score is not None else '-'}) {h.content[:60]}")
        print(f"    {'✓' if hit else '✗'} 命中关键词 {keywords}")

    print("\n[3] get_all:")
    all_records = backend.get_all_memories(user_id=user_id)
    print(f"    共 {len(all_records)} 条记忆（user_id 隔离）")
    for r in all_records[:5]:
        print(f"      - id={r.id} meta_keys={sorted(r.metadata.keys())[:6]}")

    print("\n[4] reset:")
    backend.reset_memories()
    remaining = backend.get_all_memories(user_id=user_id)
    print(f"    reset 后 {user_id} 剩余记忆: {len(remaining)}")
    ok = ok and len(remaining) == 0

    if ok:
        print("\n✓ Mem0 works (vector store: chroma, 本地持久化, 无 Docker)")
        return 0
    print("\n✗ smoke_mem0 未通过")
    return 1


if __name__ == "__main__":
    sys.exit(main())
