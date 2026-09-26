#!/usr/bin/env bash
# HugeGraph 1.7.0 单机模式安装（binary tarball，禁止 Docker/编译）：
#   1) 确保 Java 11+（缺失时 brew 安装 openjdk@17，keg-only 免 sudo）
#   2) 下载官方 tarball 到 third_party/，sha512 校验，解压后删除压缩包
#   3) 确认 RocksDB 后端配置
#   4) bin/init-store.sh 初始化
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${HUGEGRAPH_VERSION:-1.7.0}"
DIST="third_party/apache-hugegraph-incubating-${VERSION}"
TARBALL="apache-hugegraph-incubating-${VERSION}.tar.gz"
URL_MAIN="https://downloads.apache.org/incubator/hugegraph/${VERSION}/${TARBALL}"
URL_ALT="https://archive.apache.org/dist/incubator/hugegraph/${VERSION}/${TARBALL}"

# ---------- 1) Java ----------
# 注意：macOS 的 /usr/bin/java 是 stub（无 JVM 时会弹窗/挂起），不能当作可用 Java。
# HugeGraph 1.7.0 捆绑的 gremlin-groovy/Groovy 3 不支持 JDK17 字节码（class file 61），
# 因此优先 JDK 11；17 仅作未来版本兜底。
BREW_JDK11_HOME="/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
BREW_JDK17_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
JAVA_HOME_CANDIDATE=""
if [ -n "${JAVA_HOME:-}" ] && [ -x "${JAVA_HOME}/bin/java" ] && [ "${JAVA_HOME}" != "${BREW_JDK17_HOME}" ]; then
    JAVA_HOME_CANDIDATE="${JAVA_HOME}"
elif [ -x "${BREW_JDK11_HOME}/bin/java" ]; then
    JAVA_HOME_CANDIDATE="${BREW_JDK11_HOME}"
elif [ -x "${BREW_JDK17_HOME}/bin/java" ]; then
    JAVA_HOME_CANDIDATE="${BREW_JDK17_HOME}"
elif command -v java >/dev/null 2>&1 && [ "$(command -v java)" != "/usr/bin/java" ]; then
    JAVA_HOME_CANDIDATE="$(cd "$(dirname "$(command -v java)")/.." && pwd)"
fi

if [ -z "${JAVA_HOME_CANDIDATE}" ]; then
    echo "==> 未找到 Java，安装 openjdk@11（brew keg-only，免 sudo，可 brew uninstall 清除）"
    brew install openjdk@11
    JAVA_HOME_CANDIDATE="${BREW_JDK11_HOME}"
fi
export JAVA_HOME="${JAVA_HOME_CANDIDATE}"
echo "==> JAVA_HOME=${JAVA_HOME}"
"${JAVA_HOME}/bin/java" -version

# ---------- 2) 下载/解压 ----------
if [ ! -d "${DIST}" ]; then
    mkdir -p third_party
    echo "==> 下载 ${URL_MAIN}（~936MB，如慢可 HUGEGRAPH_VERSION=1.5.0 用 661MB 旧版）"
    if ! curl -fL --retry 3 -C - -o "third_party/${TARBALL}" "${URL_MAIN}"; then
        echo "==> 主镜像失败，改用 archive.apache.org"
        curl -fL --retry 3 -C - -o "third_party/${TARBALL}" "${URL_ALT}"
    fi
    # sha512 校验（从同目录下载 .sha512；失败不阻断，仅告警）
    if curl -fsSL -o "third_party/${TARBALL}.sha512" "${URL_MAIN}.sha512" \
        || curl -fsSL -o "third_party/${TARBALL}.sha512" "${URL_ALT}.sha512"; then
        if shasum -a 512 -c "third_party/${TARBALL}.sha512" --status \
            || (cd third_party && shasum -a 512 -c "${TARBALL}.sha512" --status); then
            echo "==> sha512 校验通过"
        else
            echo "⚠ sha512 校验失败（文件名不匹配时常见），跳过强校验"
        fi
    else
        echo "⚠ 未获取 .sha512，跳过校验"
    fi
    echo "==> 解压..."
    tar -xzf "third_party/${TARBALL}" -C third_party
    rm -f "third_party/${TARBALL}" "third_party/${TARBALL}.sha512"
    echo "==> 已删除 tarball（省磁盘）"
    # 三合一包中的 pd/store 子包单机模式用不到（~561MB）；HUGEGRAPH_KEEP_ALL=1 可保留
    if [ "${HUGEGRAPH_KEEP_ALL:-0}" != "1" ] && [ -d "${DIST}/apache-hugegraph-pd-incubating-${VERSION}" ]; then
        rm -rf "${DIST}/apache-hugegraph-pd-incubating-${VERSION}" \
               "${DIST}/apache-hugegraph-store-incubating-${VERSION}"
        echo "==> 已修剪 pd/store 子包（单机模式不需要；设 HUGEGRAPH_KEEP_ALL=1 可保留）"
    fi
fi
echo "==> HugeGraph 发行包: ${DIST}"

# ---------- 3) 确认 RocksDB 后端 ----------
# 1.7.0 tarball 是 pd+server+store 三合一包，server 在嵌套子目录；自动定位
if [ -f "${DIST}/bin/start-hugegraph.sh" ]; then
    SERVER="${DIST}"
else
    SERVER="${DIST}/apache-hugegraph-server-incubating-${VERSION}"
fi
PROPS="${SERVER}/conf/graphs/hugegraph.properties"
set_kv() {  # set_kv key value —— 幂等写入
    if grep -q "^${1}=" "${PROPS}" 2>/dev/null; then
        sed -i '' "s|^${1}=.*|${1}=${2}|" "${PROPS}"
    else
        echo "${1}=${2}" >>"${PROPS}"
    fi
}
set_kv backend rocksdb
set_kv serializer binary

# ---------- 3.5) gremlin 引擎修复（1.7.0 官方包两处坑，幂等） ----------
# a) gremlin-server.yaml 的 graphs 为空 → REST /gremlin 无图绑定（"No such property: g"）
GREMLIN_YAML="${SERVER}/conf/gremlin-server.yaml"
if ! grep -q "hugegraph: conf/graphs/hugegraph.properties" "${GREMLIN_YAML}"; then
    sed -i '' 's|^graphs: {$|graphs: {\n  hugegraph: conf/graphs/hugegraph.properties,|' "${GREMLIN_YAML}"
    echo "==> 已为 gremlin-server.yaml 补充 graphs 绑定"
fi
# b) scripts/empty-sample.groovy 未定义 g → 补上 graph/g 全局绑定
INIT_GROOVY="${SERVER}/scripts/empty-sample.groovy"
if ! grep -q "globals << \[g : graph.traversal()\]" "${INIT_GROOVY}"; then
    printf '\ngraph = hugegraph\nglobals << [g : graph.traversal()]\n' >>"${INIT_GROOVY}"
    echo "==> 已为 empty-sample.groovy 补充 g 绑定"
fi

echo "==> HugeGraph Server: ${SERVER}"
echo "==> 后端配置: $(grep -E '^(backend|serializer)=' "${PROPS}" | tr '\n' ' ')"

# ---------- 4) 初始化（首次或数据目录缺失时；注意 init 会清空已有数据） ----------
INIT_SCRIPT="${SERVER}/bin/init-store.sh"
if [ -d "${SERVER}/rocksdb" ] || [ -d "${SERVER}/rocksdb-data" ] || [ -d "${SERVER}/db" ]; then
    echo "==> 已有 RocksDB 数据目录，跳过 init-store（如需重置请手动执行 ${INIT_SCRIPT}）"
else
    echo "==> 初始化后端存储..."
    (cd "${SERVER}" && bash bin/init-store.sh)
fi

echo ""
echo "✓ HugeGraph 安装完成。下一步：./scripts/start_hugegraph.sh"
