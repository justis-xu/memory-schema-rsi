#!/usr/bin/env bash
# 下载 LoCoMo 官方英文版数据集（2.8MB，来自 snap-research/locomo GitHub 仓库）。
# 路径可配置：默认 data/locomo/locomo10.json，换自有中文版改 config/default.yaml。
set -euo pipefail
cd "$(dirname "$0")/.."

URL="${LOCOMO_URL:-https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json}"
DEST="${LOCOMO_DEST:-data/locomo/locomo10.json}"

mkdir -p "$(dirname "$DEST")"
if [ -s "$DEST" ]; then
    echo "已存在 $DEST（$(du -h "$DEST" | cut -f1)），跳过下载"
else
    echo "下载 $URL -> $DEST"
    curl -fL --retry 3 --progress-bar -o "$DEST.tmp" "$URL"
    mv "$DEST.tmp" "$DEST"
fi

# 校验：应为 10 个 conversation 的 JSON list（与 mem0 官方评测一致）
PY="${PYTHON_BIN:-python3}"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"
"$PY" - "$DEST" <<'EOF'
import json, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
assert isinstance(data, list), f"expected list, got {type(data).__name__}"
assert len(data) == 10, f"expected 10 conversations, got {len(data)}"
n_qa = sum(len(d.get("qa", d.get("qa_pairs", []))) for d in data)
print(f"✓ LoCoMo 校验通过: {len(data)} conversations, {n_qa} QA items, {path}")
EOF
