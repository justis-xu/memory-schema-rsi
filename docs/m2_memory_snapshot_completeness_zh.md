# 当前中文源清单完整性边界与构图变化窗口

## 问题与范围

图版本契约依赖完整的当前 Memory 清单。本阶段核两个前提：`get_all_memories` 实际能否读取全部；构图过程中源库变化是否会被发现。

范围：当前 `data/vector_store_zh/chroma/chroma.sqlite3` 的全部可见 metadata、实际导入的 Mem0/Chroma 源码、中文构图入口和后端转换。SQLite 使用只读连接及单次读事务，两次采集逻辑快照；不初始化 Chroma 客户端，不运行模型、不写真实图或记忆库。

`scripts/m2_audit_memory_snapshot_completeness.py` 与 `results/analysis/m2_memory_snapshot_completeness_20260928.json` 保存各用户计数、ID/完整metadata指纹、代码SHA和合成限额探针。

## 当前库的直接观测

可见 Memory 集合为 `schema_rsi_memories_zh`，共1,995条。另有 `mem0migrations` 集合定义，但没有进入 embeddings/metadata 记录清单。

|用户|当前条数|
|---|---:|
|zhfull:locomo:conv-26|138|
|zhfull:locomo:conv-30|102|
|zhfull:locomo:conv-41|209|
|zhfull:locomo:conv-42|195|
|zhfull:locomo:conv-43|211|
|zhfull:locomo:conv-44|186|
|zhfull:locomo:conv-47|197|
|zhfull:locomo:conv-48|224|
|zhfull:locomo:conv-49|161|
|zhfull:locomo:conv-50|177|
|nf53:locomo:conv-42|195|

中文主轨道10个用户合计1,800条，最大224条；额外195条属于另一个用户，不能混入主轨道conv-42。所有可见记录均没有非空expiration_date。两次采集所有集合metadata逻辑指纹一致。

**当前范围结论：** 单用户上限1,000远高于本次可见清单；也没有过期筛除条件。没有证据把当前S1缓存缺失、旧图残留或偏好缺关系归因于这两个边界。此结论不证明历史库从未达到上限，不证明Chroma索引与SQLite一致，也不是实际get_all客户端返回验证。

## 实际代码与合成探针

运行环境的 `Memory` 导入路径确认为仓库 `third_party/mem0-src/mem0/memory/main.py`，不是另一个安装版本。

- `Mem0Backend.get_all_memories` 对每个user调用get_all(top_k=1000)，没有分页、总量或截断状态。
- 同步Mem0 get_all 默认过滤过期记忆，底层最多取max(top_k×4,60)；这里为4,000。其 `_get_all_from_vector_store` 过滤后达到output_limit=1,000即停止。
- Chroma list调用collection.get(limit=top_k)，不补拉第二页。
- `ChromaDirectBackend` 按用户直接get，不设置limit，也不显式调用Mem0过期过滤。因此未来带过期字段时，两种后端的“当前有效源清单”可能有不同口径；当前无字段，未观测这类差异。

离线探针替换collection transport，并绕过public get_all的初始化/遥测，执行真实Mem0Backend→SDK内部过滤/格式化→Chroma list代码；fetch/output公式按上述实际public get_all构造。

|合成输入|底层limit|实际返回|
|---|---:|---:|
|1,001条有效记录|4,000|1,000|
|先4,000条过期记录，再10条有效记录|4,000|0|

合成顺序人为固定，用于证明边界；真实Chroma排序未在本轮验证。第二例说明过期过滤后的少量/空返回仍不证明库里没有更多有效记录。没有历史截断频率或答题损害结论。

## 元数据与构建期间的变化

真实 `_to_record` 合成调用保留user_id/session_date，但未把顶层expiration_date、updated_at、created_at、hash写入MemoryRecord.metadata。当前提议契约即使对转换后的全部metadata做指纹，也看不到这些字段变化。须先确定语义口径再透传：例如有效性应按明确的UTC参考日期/过期策略计算，不能只依赖隐含“今天”。

中文构图入口每个用户只在阶段开始读取一次memories，随后依次抽取、写S1、主题、聚类，末尾仅做图计数。没有重新读取源清单、对比内容指纹、源revision或发布锁。若期间删除/更新源记忆，后续仍使用开始时的对象。**这是代码推断；本轮没有观察或注入真实并发写入。**

两次SQLite快照一致只表明本轮两个采集点逻辑一致，不能排除中间写入后撤销，也不能保证未来发布后与查询间无变更。单纯“前后hash相等”是一层检测，不能冒充事务一致性；最强保证需存储revision或与写入协调的发布/读取协议。

## 结论、置信度、限制与优化建议

- **高置信度：** 本次主轨道每用户清单均小于现有限额且无过期字段，不支持当前限额截断假说。
- **高置信度，代码/合成范围：** 当前后端接口并非无条件完整枚举；达到上限时没有截断标识，过滤过期也无补页。版本契约不能凭get_all方法名接受“完整”状态。
- **高置信度，代码范围：** 构图未检测源期间变化，部分版本相关metadata在转换时被丢弃。
- **未知：** 实际历史截断、真实同时写库、线上图残留与答题收益。本阶段没有改变生产行为。

最低接入建议：

1. 源快照接口返回owner、完整性状态、总量/revision、读取口径和有效性参考日期；达到读取上限而未证明完整时拒绝图版本发布。单纯提高上限不是完整性证明。
2. 保留expiration_date/updated_at等明确来源字段，冻结提取所用输入；发布前复核同源revision。没有revision时前后完整快照对比可作临时检测，但记录其竞态限制。
3. 先保护重排前的图候选：核完整当前源ID与正文，旧ID拒绝、当前正文/日期恢复；同ID旧关系还需要关系版本验收，不能认为回填正文已全部修复。
4. 下一阶段把此源清单口径落实到最小候选准入接口，用已核旧节点病例测删除/改写与合法当前候选保护，再明确是否需要独立版本图。不要继续扩大合成场景代替中文配对收益验证。
