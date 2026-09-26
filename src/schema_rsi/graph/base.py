"""GraphStore 抽象：面向后续 Schema 演化实验的通用图存储接口。

设计约束：
- 不写死任何具体节点/边类型（V0 只有测试用的 Memory/RELATED_TO，未来会出现
  Entity/Event/State/Preference 等，由 GraphSchema 动态描述）。
- vertex 用 (label, key) 定位：key 是该 label 的业务主键属性（如 memory_id），
  upsert 语义为"存在则更新，否则创建"。
- 返回的顶点/邻居统一为 {"label", "id", "properties"} dict。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from schema_rsi.graph.schema import GraphSchema


@runtime_checkable
class GraphStore(Protocol):
    def server_info(self) -> dict:
        """服务端基本信息（版本、后端等），用于健康检查。"""
        ...

    def create_schema(self, schema: GraphSchema) -> None:
        """幂等创建 schema（propertykey/vertexlabel/edgelabel/indexlabel，已存在则跳过）。"""
        ...

    def schema_exists(self, schema: GraphSchema) -> bool:
        ...

    def drop_schema(self) -> None:
        """删除当前图的所有 schema 元素（数据应已清空）。"""
        ...

    def upsert_vertex(self, label: str, key: str, properties: dict | None = None) -> str:
        """按 (label, 主键属性=key) upsert，返回顶点 id。"""
        ...

    def upsert_edge(
        self,
        label: str,
        out_label: str,
        out_key: str,
        in_label: str,
        in_key: str,
        properties: dict | None = None,
    ) -> None:
        ...

    def get_vertex(self, label: str, key: str) -> dict | None:
        ...

    def get_neighbors(
        self,
        label: str,
        key: str,
        direction: str = "BOTH",
        edge_labels: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """1 跳邻居顶点列表。"""
        ...

    def count_vertices(self, label: str) -> int:
        ...

    def reset_graph(self, *, drop_schema: bool = True) -> None:
        """清空图数据；drop_schema=True 时连 schema 元元素一起删，回到空白图。"""
        ...

    def run_gremlin(self, query: str) -> dict:
        ...
