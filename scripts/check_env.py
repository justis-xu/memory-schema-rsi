#!/usr/bin/env python
"""环境体检：Python / 依赖 / .env / 三家 API 实测 / Java / HugeGraph / 数据文件。

用法：
    .venv/bin/python scripts/check_env.py   （或先 source .venv/bin/activate）
"""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OK, WARN, FAIL = "✓", "⚠", "✗"
results: list[tuple[str, str, str]] = []  # (status, name, detail)


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


def check_python() -> None:
    v = sys.version_info
    status = OK if (v.major, v.minor) >= (3, 11) else FAIL
    record(status, f"Python {v.major}.{v.minor}.{v.micro}", platform.platform())


def check_deps() -> None:
    for mod in ("mem0", "chromadb", "openai", "yaml", "dotenv", "requests", "pydantic"):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            record(OK, f"依赖 {mod}", ver)
        except Exception as e:
            record(FAIL, f"依赖 {mod}", str(e))


def check_env_file() -> None:
    from schema_rsi.config import PROJECT_ROOT as ROOT, get_settings

    env_path = ROOT / ".env"
    if not env_path.exists():
        record(WARN, ".env", "不存在（可 cp .env.example .env 后填写）")
        return
    s = get_settings()
    record(OK if s.llm.api_key else FAIL, ".env LLM_*",
           f"{s.llm.model} @ {s.llm.base_url}" if s.llm.api_key else "LLM_API_KEY 未填")
    record(OK if s.embedding.api_key else FAIL, ".env EMBEDDING_*",
           f"{s.embedding.model} @ {s.embedding.base_url}" if s.embedding.api_key else "EMBEDDING_API_KEY 未填")
    record(OK if s.rerank.api_key else WARN, ".env RERANK_*",
           f"{s.rerank.model}" if s.rerank.api_key else "未配置（rerank 将被跳过）")


def check_llm() -> None:
    try:
        from schema_rsi.llm.chat import make_chat_client

        client = make_chat_client()
        text, usage = client.complete(system="You are a ping responder.",
                                      user="Reply with the single word: pong", max_tokens=256)
        record(OK if "pong" in text.lower() else WARN, "LLM API 实测",
               f"{client.model} -> '{text.strip()[:30]}'")
    except Exception as e:
        record(FAIL, "LLM API 实测", str(e)[:200])


def check_embedding() -> None:
    try:
        from openai import OpenAI
        from schema_rsi.config import get_settings

        s = get_settings()
        client = OpenAI(base_url=s.embedding.base_url, api_key=s.embedding.api_key)
        resp = client.embeddings.create(model=s.embedding.model, input=["ping"])
        dim = len(resp.data[0].embedding)
        record(OK, "Embedding API 实测", f"{s.embedding.model} dim={dim}")
    except Exception as e:
        record(FAIL, "Embedding API 实测", str(e)[:200])


def check_rerank() -> None:
    try:
        from schema_rsi.config import get_settings
        from schema_rsi.llm.rerank import RerankClient

        s = get_settings()
        if not (s.rerank.api_key and s.rerank.model):
            record(WARN, "Rerank API 实测", "未配置，跳过")
            return
        client = RerankClient(s.rerank.base_url, s.rerank.api_key, s.rerank.model)
        out = client.rerank("a cat", ["The cat sleeps.", "Stocks fell today."], top_n=2)
        record(OK, "Rerank API 实测", f"{s.rerank.model} via {client.last_endpoint}, top={out[0]['index']}")
    except Exception as e:
        record(WARN, "Rerank API 实测", str(e)[:200])


def check_java() -> None:
    def java_ok() -> str | None:
        """返回可用的 JAVA_HOME 候选（处理 macOS /usr/bin/java stub 的误报）。"""
        if shutil.which("java"):
            try:
                out = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=20)
                first = (out.stderr or out.stdout).splitlines()[0] if (out.stderr or out.stdout) else ""
                if "Unable to locate" not in first:
                    return "PATH"
            except Exception:
                pass
        candidates = {
            "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home": "openjdk@17 (keg-only)",
            "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home": "openjdk@11 (keg-only)",
        }
        for home, label in candidates.items():
            if Path(home, "bin", "java").exists():
                return label
        return None

    found = java_ok()
    if found:
        record(OK, "Java", found)
    else:
        record(WARN, "Java", "未安装（运行 ./scripts/setup_hugegraph.sh 会自动装 openjdk@17）")


def check_hugegraph() -> None:
    try:
        import requests
        from schema_rsi.config import get_settings

        url = get_settings().hugegraph.url.rstrip("/")
        # HugeGraph 1.7.0 起 REST 无 /apis 前缀，老版本有
        resp = None
        for prefix in ("", "/apis"):
            try:
                resp = requests.get(f"{url}{prefix}/versions", timeout=5)
                if resp.ok:
                    record(OK, "HugeGraph", f"{url}{prefix} -> {resp.json()}")
                    return
            except Exception:
                continue
        record(WARN, "HugeGraph", "未启动；先 ./scripts/setup_hugegraph.sh && ./scripts/start_hugegraph.sh")
    except Exception as e:
        record(WARN, "HugeGraph", f"检查失败: {e}")


def check_datasets() -> None:
    from schema_rsi.config import get_settings

    s = get_settings()
    for name, path in (("LoCoMo", s.locomo_path), ("LongMemEval-S", s.longmemeval_path)):
        if not str(path):
            record(WARN, f"数据集 {name}", "路径未配置")
        elif path.exists():
            record(OK, f"数据集 {name}", f"{path} ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            record(WARN if name == "LoCoMo" else FAIL, f"数据集 {name}",
                   f"不存在: {path}（LoCoMo 运行 ./scripts/download_data.sh）")


def main() -> int:
    print("=" * 64)
    print("schema-rsi 环境体检")
    print("=" * 64)
    check_python()
    check_deps()
    check_env_file()
    check_llm()
    check_embedding()
    check_rerank()
    check_java()
    check_hugegraph()
    check_datasets()

    for status, name, detail in results:
        line = f"{status} {name}"
        if detail:
            line += f"  —  {detail}"
        print(line)
    fails = sum(1 for s, _, _ in results if s == FAIL)
    warns = sum(1 for s, _, _ in results if s == WARN)
    print("-" * 64)
    print(f"汇总: {len(results) - fails - warns} ok / {warns} warn / {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
