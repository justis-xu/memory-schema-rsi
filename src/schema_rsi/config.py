"""统一配置加载：.env（密钥/端点）+ config/default.yaml（行为参数）→ 类型化 Settings。

原则：模型相关的 base_url/api_key/model 只来自环境变量（.env / .env.example），
不在 YAML 或代码里写死厂商。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0


@dataclass
class EmbeddingConfig:
    base_url: str
    api_key: str
    model: str


@dataclass
class RerankConfig:
    base_url: str
    api_key: str
    model: str
    enabled: bool = True
    top_k: int = 5


@dataclass
class JudgeConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = False


@dataclass
class HugeGraphConfig:
    url: str = "http://127.0.0.1:8080"
    graph: str = "hugegraph"
    timeout: float = 60.0
    auth_enabled: bool = False
    username: str = "admin"
    password: str = "admin"
    jvm_xmx: str = "1g"
    dist_dir: str = (
        "third_party/apache-hugegraph-incubating-1.7.0/apache-hugegraph-server-incubating-1.7.0"
    )


@dataclass
class Settings:
    llm: LLMConfig
    embedding: EmbeddingConfig
    rerank: RerankConfig
    judge: JudgeConfig
    hugegraph: HugeGraphConfig
    mem0: dict = field(default_factory=dict)
    datasets: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    user_id_strategy: str = "per_case"
    e2e: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    # ---- convenience accessors ----
    @property
    def top_k(self) -> int:
        return int(self.mem0.get("top_k", 5))

    @property
    def graph_enabled(self) -> bool:
        return bool(self.evaluation.get("graph_enabled", False))

    @property
    def rerank_enabled(self) -> bool:
        return bool(self.rerank.enabled and self.rerank.api_key and self.rerank.model)

    @property
    def results_dir(self) -> Path:
        return self.resolve_path(self.evaluation.get("results_dir", "results"))

    @property
    def locomo_path(self) -> Path:
        return self.resolve_path(self.datasets.get("locomo_path", "data/locomo/locomo10.json"))

    @property
    def longmemeval_path(self) -> Path:
        return self.resolve_path(self.datasets.get("longmemeval_path", ""))

    def resolve_path(self, p: str | Path) -> Path:
        path = Path(os.path.expanduser(str(p)))
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def hugegraph_dist_dir(self) -> Path:
        return self.resolve_path(self.hugegraph.dist_dir)

    @property
    def baseline(self) -> dict:
        return self.raw.get("baseline") or {}

    def make_user_id(self, benchmark: str, case_id: str) -> str:
        """user_id 策略：per_case 隔离（默认）或 global 共用。"""
        if self.user_id_strategy == "global":
            return "global_user"
        return f"{benchmark}:{case_id}"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    raw = _load_yaml(Path(config_path) if config_path else DEFAULT_CONFIG_PATH)
    env = os.environ

    llm_cfg = raw.get("llm") or {}
    rerank_cfg = raw.get("rerank") or {}
    judge_cfg = raw.get("judge") or {}
    hg_cfg = raw.get("hugegraph") or {}
    hg_auth = hg_cfg.get("auth") or {}
    datasets_cfg = raw.get("datasets") or {}

    return Settings(
        llm=LLMConfig(
            base_url=env.get("LLM_BASE_URL", ""),
            api_key=env.get("LLM_API_KEY", ""),
            model=env.get("LLM_MODEL", ""),
            temperature=float(llm_cfg.get("temperature", 0.0)),
        ),
        embedding=EmbeddingConfig(
            base_url=env.get("EMBEDDING_BASE_URL", ""),
            api_key=env.get("EMBEDDING_API_KEY", ""),
            model=env.get("EMBEDDING_MODEL", ""),
        ),
        rerank=RerankConfig(
            base_url=env.get("RERANK_BASE_URL", ""),
            api_key=env.get("RERANK_API_KEY", ""),
            model=env.get("RERANK_MODEL", ""),
            enabled=bool(rerank_cfg.get("enabled", True)),
            top_k=int(rerank_cfg.get("top_k", 5)),
        ),
        judge=JudgeConfig(
            base_url=env.get("JUDGE_BASE_URL", ""),
            api_key=env.get("JUDGE_API_KEY", ""),
            model=env.get("JUDGE_MODEL", ""),
            enabled=bool(judge_cfg.get("enabled", False)),
        ),
        hugegraph=HugeGraphConfig(
            url=env.get("HUGEGRAPH_URL", hg_cfg.get("url", "http://127.0.0.1:8080")),
            graph=str(hg_cfg.get("graph", "hugegraph")),
            timeout=float(hg_cfg.get("timeout", 60)),
            auth_enabled=bool(hg_auth.get("enabled", False)),
            username=str(hg_auth.get("username", "admin")),
            password=str(hg_auth.get("password", "admin")),
            jvm_xmx=str(hg_cfg.get("jvm_xmx", "1g")),
            dist_dir=str(
                hg_cfg.get(
                    "dist_dir",
                    "third_party/apache-hugegraph-incubating-1.7.0/"
                    "apache-hugegraph-server-incubating-1.7.0",
                )
            ),
        ),
        mem0=raw.get("mem0") or {},
        datasets={
            "locomo_path": env.get(
                "LOCOMO_DATASET_PATH", datasets_cfg.get("locomo_path", "data/locomo/locomo10.json")
            ),
            "longmemeval_path": env.get(
                "LONGMEMEVAL_DATASET_PATH", datasets_cfg.get("longmemeval_path", "")
            ),
        },
        evaluation=raw.get("evaluation") or {"graph_enabled": False, "results_dir": "results"},
        user_id_strategy=str(raw.get("user_id_strategy", "per_case")),
        e2e=raw.get("e2e") or {},
        raw=raw,
    )


def reset_settings_cache() -> None:
    """测试用：清掉 lru_cache，下次 get_settings 重新读 .env/YAML。"""
    get_settings.cache_clear()
