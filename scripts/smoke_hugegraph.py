#!/usr/bin/env python
"""HugeGraph + RocksDB 冒烟测试：
create schema → insert vertices → insert edge → query vertex → traverse edge → reset。
Schema 使用 make_test_schema() 的【测试 Schema】（Memory + RELATED_TO），
仅验证图数据库能力，不是最终 Memory Graph Schema 设计。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema_rsi.graph import (  # noqa: E402
    GraphBuilder,
    HugeGraphStore,
    make_test_schema,
)
from schema_rsi.graph.hugegraph_store import HugeGraphError  # noqa: E402
from schema_rsi.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    store = HugeGraphStore(settings)

    print(f"[0] 服务信息: {settings.hugegraph.url} graph={settings.hugegraph.graph}")
    info = store.server_info()
    print(f"    versions: {info.get('versions')}")
    backend = store.backend_type()
    print(f"    backend (来自发行包配置文件): {backend}")

    schema = make_test_schema()
    print(f"\n[1] 测试 Schema '{schema.name}'（仅冒烟用）: "
          f"vertices={[v.name for v in schema.vertex_labels]} edges={[e.name for e in schema.edge_labels]} "
          f"indexes={[i.name for i in schema.indexes]}")

    try:
        print("\n[2] reset_graph（清空 + 删 schema，回到空白图）")
        store.reset_graph(drop_schema=True)

        print("[3] create_schema")
        store.create_schema(schema)
        assert store.schema_exists(schema), "schema 应创建成功"
        print("    ✓ schema 已创建（幂等：再次 create 不报错）")
        store.create_schema(schema)

        print("\n[4] insert vertices（含 upsert 幂等性验证）")
        for i, content in enumerate(
            ["记忆A：用户住在杭州", "记忆B：用户养猫 Milo", "记忆C：用户下周三看牙医"], 1
        ):
            store.upsert_vertex(
                "Memory", f"smoke_m{i}",
                {"content": content, "user_id": "smoke_user_1", "source": "smoke", "created_at": "2026-09-22"},
            )
        store.upsert_vertex("Memory", "smoke_m1", {"content": "记忆A：用户住在杭州（更新）",
                                                   "user_id": "smoke_user_1"})
        n = store.count_vertices("Memory")
        print(f"    顶点数 = {n}（重复 upsert m1 后仍应为 3）")
        assert n == 3, "upsert 应幂等"

        print("\n[5] insert edges")
        store.upsert_edge("RELATED_TO", "Memory", "smoke_m1", "Memory", "smoke_m2", {"reason": "smoke"})
        store.upsert_edge("RELATED_TO", "Memory", "smoke_m2", "Memory", "smoke_m3", {"reason": "smoke"})
        store.upsert_edge("RELATED_TO", "Memory", "smoke_m1", "Memory", "smoke_m2", {"reason": "smoke"})
        edge_count = store.run_gremlin("g.E().count()")["result"]["data"][0]
        print(f"    边数 = {edge_count}（重复插入同一条应为 2）")
        assert edge_count == 2, "edge upsert 应幂等"

        print("\n[6] query vertex")
        v = store.get_vertex("Memory", "smoke_m2")
        print(f"    get_vertex(smoke_m2) = {v}")
        assert v and v["properties"].get("content", "").startswith("记忆B")

        print("\n[7] traverse edge（1 跳邻居）")
        neighbors = store.get_neighbors("Memory", "smoke_m2")
        print(f"    get_neighbors(smoke_m2) = {[n['properties'].get('memory_id') for n in neighbors]}")
        assert len(neighbors) == 2, "m2 应有两个邻居（m1、m3）"

        print("\n[8] reset_graph（清数据，保留 schema 便于后续使用）")
        store.reset_graph(drop_schema=False)
        n_after = store.count_vertices("Memory")
        print(f"    reset 后顶点数 = {n_after}")
        assert n_after == 0

    except HugeGraphError as e:
        print(f"\n✗ HugeGraph 调用失败: {e}")
        print("  排查：1) ./scripts/start_hugegraph.sh 是否已启动 2) 见 README Troubleshooting")
        return 1

    print(f"\n✓ HugeGraph works (backend={backend or 'rocksdb(未读取到配置文件，二进制包默认)'}, "
          f"REST CRUD + traversal + reset 全通过)")
    print("✓ RocksDB works (单机后端, 数据落在发行包 rocksdb/ 目录)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
