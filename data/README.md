# data/ —— 基准数据集放置说明

数据文件不入 git（见 .gitignore），按需放置：

## LoCoMo

```bash
./scripts/download_data.sh        # 下载官方英文版 locomo10.json (2.8MB) 到 data/locomo/
```

后续如需换自己的中文版：把文件放到任意路径，改 `config/default.yaml` 的
`datasets.locomo_path` 或设环境变量 `LOCOMO_DATASET_PATH`。不修改数据内容。

## LongMemEval-S

当前 `config/default.yaml` 直接引用本机已有文件（不复制，节省磁盘）：

```
/Users/xu/git/eval-datasets/longmemeval-zh/longmemeval_s_cleaned.json   (英文原版, 265MB)
```

（历史位置 memory-prompt/longmemeval-zh 已迁移；若再迁移只需改 `datasets.longmemeval_path`。）

换中文版：改 `datasets.longmemeval_path` 或设 `LONGMEMEVAL_DATASET_PATH`。

注意：这是原版 LongMemEval-S（cleaned 版），不是 LongMemEval-V2。
加载该 JSON 的进程峰值内存约 2~2.5GB（Python 对象膨胀），16GB 机器无压力。
