# LifecycleRSI验收完整性：未知结果不能计作零反伤

## 问题与范围

本阶段转查实验RSI的实际候选选择/验收接口。先读既有 `docs/rsi_evidence_audit_zh.md` 与验证器支持ID专项，避免重复既有英文/代理效果统计。

新问题：`LifecycleRSI` 接受 injected evaluator 的完整题集后，是否保证每道题都有可比较判分和真实成本？源支持ID与此判分/成本完整性是不同门槛。

代码范围 `experiments/schema_rsi/schema_rsi_lab/lifecycle_rsi.py`；离线7种控制，每种5个独立conversation、每conversation一题基线错/一题基线对，共10题。使用真实LifecycleRSI.run，evaluator为故障注入桩，所有输出目录在临时目录。零模型调用，未改主评测/Memory/Graph/Jev数据。

脚本 `scripts/m2_audit_lifecycle_verdict_contract.py`；before/after机器记录 `results/analysis/m2_lifecycle_verdict_contract_{before,after}_20260928.json` 保留各自代码SHA，不覆盖旧运行。

## 实际接口与新发现

LifecycleRSI按conversation分train/validation/test；5个候选在train排序，只将前2个送validation，第一满足fixed≥1、harmed=0、mean_model_calls≤3的候选晋升，再读test。它已核返回ID集合、重复ID、同次配对baseline判分未改变。

但旧_summary只用`correct is True/False`计数，缺失或未知值不能计为正确，也不会触发`correct is False and baseline_correct is True`的反伤；model_calls/edge_visits缺失又默认为0，负数也可转换并累计。

离线unknown_verdict情形：validation两题，原错题correct=True修复1；原对题correct=None尚未有明确结果。旧报告给accuracy=0.5、harmed=0、net=1，候选accepted=True。这不是证明原对题真被答错，而是**缺失验收证据被允许通过“零反伤”条件**。

|注入条件|旧行为|修复后|
|---|---|---|
|完整有效结果/成本|晋升|仍晋升|
|明确反伤False|拒绝晋升|仍拒绝|
|correct=None|晋升，未知未计反伤|拒绝评估|
|correct="false"字符串|晋升，未计反伤|拒绝评估|
|缺model_calls|晋升，均值0|拒绝评估|
|model_calls=-1|晋升，均值-1|拒绝评估|
|缺edge_visits|晋升，均值0|拒绝评估|

成本缺失不等于实际调用0，负数不合法；这些合成成本不代表真实服务调用情况。

## 修复与验证

仅收紧实验_summary：correct与baseline_correct必须是严格bool；model_calls与edge_visits必须显式提供、类型为int且非负（排除bool伪整数），再计算结果。未知判分应作为incomplete trial处理，不能压入已完成的晋升评估；零调用/零访问仍允许，但须明确记录。

七种真实run控制的before/after预期断言全部符合；`.venv/bin/python -m pytest experiments/schema_rsi/tests/test_lifecycle.py -q`为7 passed。代码编译及diff检查通过。该环境当前有pytest；不沿用较早“pytest不存在”的环境记录。

本改变可能使旧adapter因缺成本字段而失败，调用者应补真实账，不能为兼容而填伪0。现有测试用合法整数账，回归通过；没有已核的生产LifecycleRSI运行受此改变影响。

## 与历史schema演化归档的区别

查到两份 `experiments/schema_rsi/runs/run-20260923T000107786191Z` 与 `...000649010998Z` 的impact文件。它们是EvolutionEngine，并非LifecycleRSI：该引擎从train utility排序取前8个，validation按平均evidence-ID召回delta、访问与弧密度、相对active utility门控，最后冻结后读test。不能把本修复或LifecycleRSI的harmed=0条件套在那两份历史代理结果上。

既有报告已界定两份为6合成case的召回代理，本阶段没有重算效果，也没有发现其实际判分None的证据。EvolutionEngine的validation反复参与候选接受，test直到冻结才读；“validation gate”不等于完全未用于策略选择的最终泛化集。

## 结论、置信度、限制与建议

**高置信度，代码/离线控制：** LifecycleRSI曾将未知判分与未知成本混入完成评估，允许假定零反伤/低成本的晋升；现在不完整trial不能进入_summary。只保证评估字段与账完整，不保证判分真实或source关系有效。

**未知：** 真实历史或当前模型产生这些值的频率、拒绝后净收益、中文闭环学习收益。没有给旧答案改分，没有证明历史晋升错误，也没把合成10题当中文样本。

下一步：

1. 实验adapter分别保留completed/failed/ungraded、严格答案标签、来源支持与调用账；只有完整成对证据进入晋升。记录服务失败而不是默认成错误或零成本。
2. 继续核EvolutionEngine按平均召回utility接受时是否会掩盖逐题证据退化，区别它与LifecycleRSI的零反伤策略；代理召回改善与答案改善分开。
3. 真实中文闭环仍需要无当前gold的经验产生、独立后续题与相同调用预算。支持编号、字段完整和schema冻结都只是必要条件，不能代替因果收益证明。
