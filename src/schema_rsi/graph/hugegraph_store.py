"""HugeGraph REST API 适配层（单机模式 + RocksDB 后端）。

- 仅依赖 requests，不引入旧版 pyhugegraph / Java client。
- schema 创建幂等（先 GET 判存在再 POST）。
- 顶点以 (label, 业务主键属性值) 定位：条件查询依赖对应 secondary index。
- clear 接口按 HugeGraph 要求携带 confirm_message。
- 可选鉴权：hugegraph.auth.enabled=true 时走 /apis/login 换 token。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from schema_rsi.config import HugeGraphConfig, Settings, get_settings
from schema_rsi.graph.schema import GraphSchema

logger = logging.getLogger(__name__)

CLEAR_CONFIRM_MESSAGE = "I'm sure to delete all data"


class HugeGraphError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class HugeGraphStore:
    def __init__(self, settings: Settings | None = None, config: HugeGraphConfig | None = None):
        self.settings = settings or get_settings()
        self.cfg = config or self.settings.hugegraph
        self.graph = self.cfg.graph
        self._session = requests.Session()
        self._token: str | None = None
        # HugeGraph 1.7.0 起 REST 无 /apis 前缀；老版本在 /apis 下 —— 自动探测
        self.base = self.cfg.url.rstrip("/") + self._detect_prefix()
        if self.cfg.auth_enabled:
            self._login()
        # label -> 业务主键属性名（由 create_schema 记录；跨进程持久化到本地文件）
        self._key_props: dict[str, str] = {}
        self._key_props_file = self.settings.resolve_path("data/graph/key_props.json")
        self._key_props.update(self._load_key_props())
        # 最近一次 create_schema 的 schema（/clear 会连 schema 一起清掉，
        # reset_graph(drop_schema=False) 需要用它重建）
        self._last_schema: GraphSchema | None = None

    def _detect_prefix(self) -> str:
        root = self.cfg.url.rstrip("/")
        for prefix in ("", "/apis"):
            try:
                resp = self._session.get(f"{root}{prefix}/versions", timeout=5)
                if resp.ok:
                    return prefix
            except requests.RequestException:
                continue
        return ""  # 服务未启动时默认新版本路径；真正调用时会给出清晰报错

    # ---------------- low-level ----------------

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _login(self) -> None:
        resp = requests.post(
            f"{self.base}/login",
            json={"name": self.cfg.username, "password": self.cfg.password},
            timeout=self.cfg.timeout,
        )
        if resp.ok and resp.json().get("token"):
            self._token = resp.json()["token"]
        else:
            logger.warning("HugeGraph login failed: %s %s", resp.status_code, resp.text[:200])

    def _request(self, method: str, path: str, *, params: dict | None = None, body: Any = None) -> Any:
        url = f"{self.base}{path}"
        data = None if body is None else json.dumps(body)
        try:
            resp = self._session.request(
                method, url, params=params, data=data, headers=self._headers(), timeout=self.cfg.timeout
            )
        except requests.RequestException as e:
            raise HugeGraphError(f"HugeGraph request error ({method} {path}): {e}") from e
        if resp.status_code == 401 and self._token is None and self.cfg.username:
            # 服务端开启了鉴权而本地未配置：尝试登录一次后重试
            self._login()
            if self._token:
                return self._request(method, path, params=params, body=body)
        if not resp.ok:
            raise HugeGraphError(
                f"HugeGraph {method} {path} -> HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                body=resp.text[:500],
            )
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    # ---------------- GraphStore 协议实现 ----------------

    def server_info(self) -> dict:
        info: dict[str, Any] = {}
        try:
            info["versions"] = self._request("GET", "/versions")
        except HugeGraphError as e:
            info["versions_error"] = str(e)
        try:
            info["graphs"] = self._request("GET", "/graphs")
        except HugeGraphError as e:
            info["graphs_error"] = str(e)
        if isinstance(info.get("graphs"), list) and self.graph in info["graphs"]:
            info["graph_detail"] = self._request("GET", f"/graphs/{self.graph}")
        return info

    # ---- schema ----

    def _schema_exists(self, kind: str, name: str) -> bool:
        try:
            self._request("GET", f"/graphs/{self.graph}/schema/{kind}/{name}")
            return True
        except HugeGraphError as e:
            if e.status_code == 404:
                return False
            raise

    def _create_if_missing(self, kind: str, name: str, body: dict) -> bool:
        if self._schema_exists(kind, name):
            return False
        self._request("POST", f"/graphs/{self.graph}/schema/{kind}", body=body)
        return True

    def create_schema(self, schema: GraphSchema) -> None:
        self._last_schema = schema
        # 1) property keys（全部去重合并）
        all_props: dict[str, tuple[str, str]] = {}
        for vl in schema.vertex_labels:
            for p in vl.properties:
                all_props[p.name] = (p.pg_type, p.cardinality)
        for el in schema.edge_labels:
            for p in el.properties:
                all_props.setdefault(p.name, (p.pg_type, p.cardinality))

        for name, (pg_type, cardinality) in all_props.items():
            self._create_if_missing(
                "propertykeys",
                name,
                {
                    "name": name,
                    "data_type": pg_type.upper(),
                    "cardinality": cardinality.upper(),  # SINGLE/SET/LIST
                },
            )

        # 2) vertex labels（本层固定映射为 AUTOMATIC id + 业务主键属性上的 secondary index）
        for vl in schema.vertex_labels:
            if vl.primary_key:
                self._key_props[vl.name] = vl.primary_key
            self._create_if_missing(
                "vertexlabels",
                vl.name,
                {
                    "name": vl.name,
                    "id_strategy": "AUTOMATIC",
                    "properties": vl.property_names(),
                    "nullable_keys": vl.property_names(),
                    "enable_label_index": True,
                },
            )
        self._save_key_props()

        # 3) edge labels
        for el in schema.edge_labels:
            source = el.source_labels[0] if len(el.source_labels) == 1 else ""
            target = el.target_labels[0] if len(el.target_labels) == 1 else ""
            self._create_if_missing(
                "edgelabels",
                el.name,
                {
                    "name": el.name,
                    "source_label": source,
                    "target_label": target,
                    "properties": el.property_names(),
                    "nullable_keys": el.property_names(),
                    "enable_label_index": True,
                },
            )

        # 4) index labels（字段名：base_type=VERTEX/EDGE, base_value=label 名, fields=属性列表）
        for idx in schema.indexes:
            self._create_if_missing(
                "indexlabels",
                idx.name,
                {
                    "name": idx.name,
                    "base_type": "VERTEX_LABEL",
                    "base_value": idx.label,
                    "fields": [idx.field],
                    "index_type": idx.index_type.upper(),
                },
            )

    def schema_exists(self, schema: GraphSchema) -> bool:
        for vl in schema.vertex_labels:
            if not self._schema_exists("vertexlabels", vl.name):
                return False
        for el in schema.edge_labels:
            if not self._schema_exists("edgelabels", el.name):
                return False
        for idx in schema.indexes:
            if not self._schema_exists("indexlabels", idx.name):
                return False
        return True

    def drop_schema(self) -> None:
        g = self.graph
        # 顺序：indexlabel -> edgelabel -> vertexlabel -> propertykey
        for kind in ("indexlabels", "edgelabels", "vertexlabels"):
            names = self._list_schema(kind)
            for name in names:
                try:
                    self._request("DELETE", f"/graphs/{g}/schema/{kind}/{name}")
                except HugeGraphError as e:
                    logger.warning("drop %s/%s failed: %s", kind, name, e)
        # propertykey 可能仍被引用，逐个尝试
        for name in self._list_schema("propertykeys"):
            try:
                self._request("DELETE", f"/graphs/{g}/schema/propertykeys/{name}")
            except HugeGraphError as e:
                logger.warning("drop propertykeys/%s failed: %s", name, e)

    def _list_schema(self, kind: str) -> list[str]:
        try:
            data = self._request("GET", f"/graphs/{self.graph}/schema/{kind}")
        except HugeGraphError as e:
            logger.warning("list %s failed: %s", kind, e)
            return []
        return [item["name"] for item in (data or []) if isinstance(item, dict) and "name" in item]

    # ---- vertices ----

    def _load_key_props(self) -> dict[str, str]:
        try:
            import json as _json

            data = _json.loads(self._key_props_file.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()}
        except (OSError, ValueError):
            return {}

    def _save_key_props(self) -> None:
        self._key_props_file.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        self._key_props_file.write_text(
            _json.dumps(self._key_props, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _key_prop(self, label: str) -> str:
        kp = self._key_props.get(label)
        if not kp:
            raise HugeGraphError(
                f"label '{label}' 的业务主键未知：请先在本 store 实例上调用 create_schema(schema)"
            )
        return kp

    def _query_vertices(self, label: str, props: dict, limit: int = 10) -> list[dict]:
        data = self._request(
            "GET",
            f"/graphs/{self.graph}/graph/vertices",
            params={"properties": json.dumps(props), "limit": limit},
        )
        vertices = (data or {}).get("vertices", []) if isinstance(data, dict) else []
        return [v for v in vertices if v.get("label") == label]

    def upsert_vertex(self, label: str, key: str, properties: dict | None = None) -> str:
        kp = self._key_prop(label)
        props = {kp: str(key)}
        for k, v in (properties or {}).items():
            if v is not None:
                props[k] = str(v)
        existing = self._query_vertices(label, {kp: str(key)}, limit=2)
        if existing:
            vid = existing[0]["id"]
            updated = self._request(
                "PUT",
                f"/graphs/{self.graph}/graph/vertices/{vid}",
                params={"action": "append"},  # 顶点更新 action（小写 append/eliminate）
                body={"label": label, "properties": props},
            )
            return (updated or {}).get("id", vid)
        # 此端点收单个对象（非数组）
        created = self._request(
            "POST", f"/graphs/{self.graph}/graph/vertices", body={"label": label, "properties": props}
        )
        items = created if isinstance(created, list) else [created]
        return items[0]["id"]

    def get_vertex(self, label: str, key: str) -> dict | None:
        kp = self._key_prop(label)
        found = self._query_vertices(label, {kp: str(key)}, limit=2)
        if not found:
            return None
        v = found[0]
        return {"label": v.get("label"), "id": v.get("id"), "properties": v.get("properties", {})}

    # ---- edges ----

    def upsert_edge(
        self,
        label: str,
        out_label: str,
        out_key: str,
        in_label: str,
        in_key: str,
        properties: dict | None = None,
    ) -> None:
        out_v = self.get_vertex(out_label, out_key)
        in_v = self.get_vertex(in_label, in_key)
        if out_v is None or in_v is None:
            raise HugeGraphError(
                f"upsert_edge: 顶点不存在 out=({out_label},{out_key}) in=({in_label},{in_key})"
            )
        out_id, in_id = out_v["id"], in_v["id"]
        # 幂等：已存在同 label + 同两端的边则跳过
        data = self._request(
            "GET",
            f"/graphs/{self.graph}/graph/edges",
            params={"vertex_id": out_id, "direction": "OUT", "limit": 200},
        )
        edges = (data or {}).get("edges", []) if isinstance(data, dict) else []
        for e in edges:
            if e.get("label") == label and e.get("outV") == out_id and e.get("inV") == in_id:
                return
        props = {k: str(v) for k, v in (properties or {}).items() if v is not None}
        self._request(
            "POST",
            f"/graphs/{self.graph}/graph/edges",
            body={
                "label": label,
                "outV": out_id,
                "inV": in_id,
                "outVLabel": out_label,   # 1.7.0 为驼峰命名
                "inVLabel": in_label,
                "properties": props,
            },
        )

    def get_neighbors(
        self,
        label: str,
        key: str,
        direction: str = "BOTH",
        edge_labels: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        center = self.get_vertex(label, key)
        if center is None:
            return []
        vid = center["id"]
        data = self._request(
            "GET",
            f"/graphs/{self.graph}/graph/edges",
            params={"vertex_id": vid, "direction": direction, "limit": limit * 2},
        )
        edges = (data or {}).get("edges", []) if isinstance(data, dict) else []
        neighbors: dict[Any, dict] = {}
        for e in edges:
            if edge_labels and e.get("label") not in edge_labels:
                continue
            other = e.get("inV") if e.get("outV") == vid else e.get("outV")
            if other is None or other == vid or other in neighbors:
                continue
            if len(neighbors) >= limit:
                break
            try:
                v = self._request("GET", f"/graphs/{self.graph}/graph/vertices/{other}")
            except HugeGraphError:
                continue
            neighbors[other] = {
                "label": v.get("label"),
                "id": v.get("id"),
                "properties": v.get("properties", {}),
                "via_edge": e.get("label"),
            }
        return list(neighbors.values())

    # ---- stats / maintenance ----

    def count_vertices(self, label: str) -> int:
        data = self.run_gremlin(f"g.V().hasLabel('{label}').count()")
        results = (data or {}).get("result", {}).get("data", [])
        return int(results[0]) if results else 0

    def reset_graph(self, *, drop_schema: bool = True) -> None:
        # HugeGraph clear 必须以 query 参数携带 confirm_message（1.7.0 的确认串为
        # "I'm sure to delete all data"；若版本不同会返回 400 并提示期望值，自适应重试一次）
        def _clear(confirm: str):
            return self._session.delete(
                f"{self.base}/graphs/{self.graph}/clear",
                params={"confirm_message": confirm},
                headers=self._headers(),
                timeout=self.cfg.timeout,
            )

        resp = _clear(CLEAR_CONFIRM_MESSAGE)
        if resp.status_code == 400 and "Please take the message:" in resp.text:
            expected = resp.text.split("Please take the message:", 1)[1].strip('"')
            resp = _clear(expected)
        if not resp.ok:
            raise HugeGraphError(
                f"HugeGraph clear -> HTTP {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )
        # 注意：HugeGraph 的 /clear 连 schema 元素一起清除
        if drop_schema:
            self.drop_schema()
            self._key_props.clear()
            self._save_key_props()
            self._last_schema = None
        elif self._last_schema is not None:
            # 语义：清数据但保留 schema —— 用最近一次的 schema 重建
            self.create_schema(self._last_schema)

    def run_gremlin(self, query: str) -> dict:
        # 不显式传 language（1.7.0 传 "gremlin-groovy" 反而报 ScriptEngine 不可用）
        return self._request("POST", "/gremlin", body={"gremlin": query})

    def backend_type(self) -> str | None:
        """从本机发行包的 hugegraph.properties 读后端类型（服务端 API 不暴露此项）。"""
        props_file = self.settings.hugegraph_dist_dir / "conf" / "graphs" / f"{self.graph}.properties"
        try:
            for line in props_file.read_text().splitlines():
                if line.strip().startswith("backend="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            pass
        return None
