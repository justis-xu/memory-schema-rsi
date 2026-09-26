#!/usr/bin/env bash
# 停止 HugeGraph Server。
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${HUGEGRAPH_VERSION:-1.7.0}"
DIST="${HUGEGRAPH_DIST:-third_party/apache-hugegraph-incubating-${VERSION}}"
URL="${HUGEGRAPH_URL:-http://127.0.0.1:8080}"

# 1.7.0 tarball 是三合一包，server 在嵌套子目录；自动定位
if [ -f "${DIST}/bin/stop-hugegraph.sh" ]; then
    SERVER="${DIST}"
else
    SERVER="${DIST}/apache-hugegraph-server-incubating-${VERSION}"
fi

hg_alive() {
    curl -sf -m 3 "${URL}/versions" >/dev/null 2>&1 \
        || curl -sf -m 3 "${URL}/apis/versions" >/dev/null 2>&1
}

if ! hg_alive; then
    echo "HugeGraph 未在运行"
    exit 0
fi

STOP_SCRIPT="bin/stop-hugegraph.sh"
[ -f "${SERVER}/${STOP_SCRIPT}" ] || STOP_SCRIPT="bin/stop-hugeserver.sh"

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
fi

(cd "${SERVER}" && bash "${STOP_SCRIPT}")

for i in $(seq 1 15); do
    if ! hg_alive; then
        echo "✓ HugeGraph 已停止"
        exit 0
    fi
    sleep 2
done
echo "⚠ 端口仍开放，可能未完全停止；可手动检查: lsof -nP -iTCP:8080 -sTCP:LISTEN"
exit 0
