# 单对话中文写入入口：可选持久化来源与返回正文账

> 2026-09-29。只改 `scripts/ingest_conv.py` 的显式可选路径和 `source_trace.py` 的 JSONL sink；离线验证 `scripts/m2_verify_ingest_conv_trace.py`，机器结果 `results/analysis/m2_ingest_conv_trace_20260929.json`。没有实际 Mem0/Chroma/HugeGraph 写入或模型调用。

## 为什么需要入口

已有 `EvaluationPipeline.ingest_case(source_trace_sink=...)` 能保存输入批次、运行/尝试身份及本次 backend 返回正文，但正常单对话写入脚本没有传 sink。只在接口层支持留痕，真正跑一次中文写入时仍会丢失这条事件链。本阶段让 `scripts/ingest_conv.py` 在显式传 `--trace-output PATH` 时独占创建 JSONL 文件；不传该选项时仍走原调用。没有改全量评测入口或生产图、检索配置。

可选账先写 `run_started`，记录本次 `run_id`、尝试 ID、case/user、中文源文件和配置文件 SHA256，并记录**生效提取配置的安全子集与指纹**：模型名、温度、最大输出、关闭思考开关、提示词指纹、embedding 模型、集合名及端点指纹；不写密钥或原始端点。每个场次紧接 `batch_input`、`batch_outcome`；结束写 `run_completed`，含新增记录数、写入错误数、再次计算的源文件 SHA 和“期间未变化”布尔值。`run_id` 未指定时自动生成 UUID，`attempt_id` 默认 `cli-1`。两个批次事件保留相同身份；`batch_id` 仍只代表输入内容。

事件逐行写入、flush 并同步到磁盘。已有文件通过独占创建拒绝覆盖，且在 backend 调用前失败。若中途异常导致没有 `run_completed`，文件代表**不完整尝试**；尤其结果事件写入失败时 backend 可能已写，不能据文件尾部缺失盲目重跑。数据文件在加载前后若变更，则在写入前终止；加载后到结束若变更，会在完成事件中标明。输入事件本身保存实际进入 backend 的消息，因此文件指纹变化也不能抹去该批具体输入。

示例（只有显式执行才会实际写入）：

```bash
.venv/bin/python scripts/ingest_conv.py --config config/locomo_zh.yaml \
  --prefix isolated-study --conv conv-42 \
  --trace-output results/analysis/m2_conv42_source_trace.jsonl
```

注意该脚本本身仍是实际 Mem0 写入入口。上述命令仅展示用法，本阶段**没有执行**。若目标 user_id 已有记忆，原脚本会直接跳过，本可选账也不会生成；因此不能用“脚本运行成功”代替实际提取完成。JSONL 含原始对话和返回记忆正文，研究时应按数据集产物管理。

## 验证、结论与边界

测试用一场中文控制输入核开关前后 backend 请求一致、事件四步齐、正文与指纹可读、gold 不进入来源事件，并验证已有目标文件会在 backend 前拒绝、配置快照不带密钥和原始端点；相关测试合计 **14 passed**。另用仓库已归档的工程师题两场真实中文原话和 Recorder 注入的上一阶段 POC 事实，得到 `run_started → 两对 batch_input/batch_outcome → run_completed` 共六个事件；源文件 SHA 未变化、两条正文按场次配对，开关前后请求相同。注入文本不是历史 Mem0 实际输出。

**结论：** 对单对话小研究，来源账现在可由入口显式开启并持久化，输入批次和当次返回文本有可复核的同运行事件链。它并未自动拆事实、判断 full/partial/absent，也不保证返回列表就是写入后的完整库存。默认评测运行仍未启用此功能。

**置信度：** 对离线控制、文件独占、事件顺序、请求一致性和测试为高；真实 Mem0 长对话写入、失败恢复、自动语义支持及答题净收益未知。

**下一步优化建议：** 真正做隔离的中文小提取时，先冻结源文件、配置、模型和 user_id，控制预计调用量，并显式启用本账；完成后对照写入后完整库存与 `returned_records`，逐子句人工核源、遗漏、更新时间和反伤。全量入口只有在这条小路径通过后再考虑接入，不能因事件可写就扩大模型调用。
