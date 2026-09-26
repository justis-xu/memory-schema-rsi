#!/usr/bin/env bash
# 启动 HugeGraph Server（单机 + RocksDB），显式限制 JVM 堆（默认 -Xmx1g，可 HUGEGRAPH_XMX 覆盖）。
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${HUGEGRAPH_VERSION:-1.7.0}"
DIST="${HUGEGRAPH_DIST:-third_party/apache-hugegraph-incubating-${VERSION}}"
XMX="${HUGEGRAPH_XMX:-1g}"
URL="${HUGEGRAPH_URL:-http://127.0.0.1:8080}"

# 健康检查：1.7.0 无 /apis 前缀，老版本有 —— 两个都试
hg_alive() {
    curl -sf -m 3 "${URL}/versions" >/dev/null 2>&1 \
        || curl -sf -m 3 "${URL}/apis/versions" >/dev/null 2>&1
}

# Java 定位（与 setup_hugegraph.sh 同逻辑；/usr/bin/java 是 macOS stub，不算可用。
# HugeGraph 1.7.0 的 gremlin 引擎需要 JDK 11：Groovy 3 不支持 JDK17 字节码）
BREW_JDK11_HOME="/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
BREW_JDK17_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
if [ -x "${BREW_JDK11_HOME}/bin/java" ]; then
    export JAVA_HOME="${BREW_JDK11_HOME}"
elif [ -n "${JAVA_HOME:-}" ] && [ -x "${JAVA_HOME}/bin/java" ]; then
    :
elif [ -x "${BREW_JDK17_HOME}/bin/java" ]; then
    export JAVA_HOME="${BREW_JDK17_HOME}"
elif command -v java >/dev/null 2>&1 && [ "$(command -v java)" != "/usr/bin/java" ]; then
    export JAVA_HOME="$(cd "$(dirname "$(command -v java)")/.." && pwd)"
else
    echo "✗ 未找到 Java，先运行 ./scripts/setup_hugegraph.sh" >&2
    exit 1
fi

if [ ! -d "${DIST}" ]; then
    echo "✗ 未找到 ${DIST}，先运行 ./scripts/setup_hugegraph.sh" >&2
    exit 1
fi

# 1.7.0 tarball 是 pd+server+store 三合一包，server 在嵌套子目录；自动定位
if [ -f "${DIST}/bin/start-hugegraph.sh" ]; then
    SERVER="${DIST}"
else
    SERVER="${DIST}/apache-hugegraph-server-incubating-${VERSION}"
fi

if hg_alive; then
    echo "HugeGraph 已在运行：${URL}"
    exit 0
fi

# 官方 start-hugegraph.sh 通过 -j 传 USER_OPTION（JVM 参数），自带 daemon + 就绪等待
echo "==> 启动 HugeGraph（${SERVER}，-Xmx${XMX}）..."
(cd "${SERVER}" && bash bin/start-hugegraph.sh -j "-Xms256m -Xmx${XMX}" -t 60) || true

echo -n "==> 验证 ${URL}"
for i in $(seq 1 30); do
    if hg_alive; then
        echo ""
        echo "✓ HugeGraph works: $(curl -sf "${URL}/versions" || curl -sf "${URL}/apis/versions")"
        echo "  停止: ./scripts/stop_hugegraph.sh"
        exit 0
    fi
    echo -n "."
    sleep 2
done
echo ""
echo "✗ 服务未就绪。查看日志："
tail -n 30 "${SERVER}"/logs/*.log 2>/dev/null || true
exit 1
