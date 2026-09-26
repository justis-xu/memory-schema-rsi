#!/usr/bin/env bash
# 从 Hugging Face 私有数据仓库拉取评测产物（results/、results_zh/、experiments/schema_rsi/runs/，
# 约 690MB）。用途：换机分析——git 仓库只含代码与报告，JSONL 原始证据存 HF。
# 仓库为 dataset 类型，文件按项目内相对路径还原，拉完即用。
#
# 用法：
#   ./scripts/pull_results.sh                      # 全量（~690MB）
#   INCLUDE='results_zh/*' ./scripts/pull_results.sh   # 只拉中文轨道
#   HF_MIRROR=1 ./scripts/pull_results.sh          # 国内走 hf-mirror 镜像
#
# 可配置环境变量：
#   HF_REPO   仓库 id，默认 justis-xu/memory-schema-rsi-data
#   INCLUDE   传给 --include 的通配模式（如 'results/*'）
#   HF_MIRROR =1 时设置 HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${HF_REPO:-justis-xu/memory-schema-rsi-data}"
[ "${HF_MIRROR:-0}" = "1" ] && export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if ! command -v hf >/dev/null 2>&1; then
    echo "缺少 hf CLI：uv pip install -U huggingface_hub（或 pip install -U huggingface_hub）" >&2
    exit 1
fi
if ! hf auth whoami >/dev/null 2>&1; then
    echo "未登录 HF：hf auth login（token 见 https://huggingface.co/settings/tokens，read 权限即可）" >&2
    exit 1
fi

ARGS=(--repo-type dataset --local-dir .)
[ -n "${INCLUDE:-}" ] && ARGS+=(--include "$INCLUDE")

# 断点续传：已存在且哈希一致的文件自动跳过，中断后重跑同命令即可
hf download "$REPO" "${ARGS[@]}"

echo "✓ 拉取完成，文件已按项目相对路径还原（结论摘要见 results*/*_report.md）"
