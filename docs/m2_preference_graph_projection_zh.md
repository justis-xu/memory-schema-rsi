# 偏好来源限定进入S1 Graph时的字段与缓存边界（2026-09-28）

## 问题与方法

中文Memory中的赞同/提问限定已出现缺失，后续若给提取结果增加来源字段，当前S1图是否保留并使用？现有图缓存是否覆盖这两个病例的当前Memory？

固定检查吉娜/乔恩舞风和内特游戏相关4个记忆ID；当前conv30/conv42的102/195条快照按SQLite只读重新核ID、正文；读取 `data/graph_zh/structured_cache.json`，使用真实 `StructuredExtractor._normalize`、`GraphBuilder.build_s1` 和本地记录器回放。记录器只在内存保存节点/边，不连接真实图服务。另构造2条合成结构记录检验同键偏好合并，只验证代码契约，不代表自动模型输出或benchmark反伤。

脚本 `scripts/m2_audit_preference_graph_projection.py`；机器结果 `results/analysis/m2_preference_graph_projection_20260928.json`，保存缓存条目、节点/边、合成原字段/归一化结果及代码SHA。零模型/embedding/rerank/答题/裁判请求，未写Graph或Memory。反伤核查是避免把缺缓存当空关系、把合成字段合并当真实失分、把原记忆链接仍在误说成全部来源丢失。

## 实际缓存与版本

| 当前用户 | Memory数量 | 当前ID命中中文structured缓存 | 边界 |
| --- | --- | --- | --- |
| `zhfull:locomo:conv-30` |102|80，另22缺|乔恩最爱 `e093b97c-…`和吉娜表演 `7063f8d7-…`有缓存，不能据此称全对话当前S1齐备。 |
| `zhfull:locomo:conv-42` |195|0，全部195缺|缓存有旧 `0877eae7-…`，没有当前 `c6aaa7b8-…`。旧缓存不是当前完整构图依据。 |

这只是指定缓存文件与当前两个用户ID的对应账；没有读取真实Graph服务，因此不能说线上图一定缺195条、或其schema一定是S1。缓存存在也没有输入正文/模型/提示指纹来保证生成条件相同。本轮重新只读核当前ID/正文与两份Memory审计快照一致。

4条选定Memory的本地回放生成4个Memory、7个Entity、4个Preference、0个Event。缓存给乔恩 `like all dance styles` 和 `prefer modern dance`，吉娜团队表演没有Preference；旧内特给 `like Xenoblade Chronicles` 与 `like Nintendo games`。当前内特ID无缓存，本回放仍生成其Memory节点，但不生成其派生实体/偏好——这是把缺缓存当空结构输入的模拟结果，不是真实图状态或重新自动提取。

## 字段如何投影

`extractor_s1.py` 提示输入只有 `[memory_id] content`，不带原始对话邻轮。Preference约定为subject/pref_type/object，pref_type建议like/dislike/prefer/avoid；`_normalize`也只保留这3个字段。即使模型返回source_turn_ids、support_strength、source_date，当前归一化会删除它们。

`builder.py` 偏好键是用户作用域中的 `subject|pref_type|object`，节点只写这3个语义字段与pref_id；Memory通过EXPRESSES_PREFERENCE连接节点，人实体通过HAS_PREFERENCE连接节点。Schema无逐事实源轮、支持方式、适用时间或强度专用字段。`pref_type`字符串可表达prefer等差异；代码没有严密枚举验证，因此不能断言任何额外类型绝对无法出现，但也没有来源强度的明确契约。

合成对照故意给同一subject/type/object两种support_strength：explicit_favorite和recently_read，分别带S1/S2源轮。真实归一化后两条都只剩相同3字段，本地构图合成**1个Preference节点，2条Memory到Preference边**。这证明当前同键投影不会保留额外强度字段；仍保留两个Memory链接，不能称全部来源丢失。合成类型固定为like，并不证明真实模型把所有最爱与近期阅读都自动输出同一个类型。

## 检索消费与意义

当前 `GraphRetriever._expand_s1` 的Preference-hop从锚点Memory出发，经EXPRESSES_PREFERENCE到Preference，再沿同边反向取关联Memory，使用 `pref:<object>` 记录via。返回的是Memory正文/元数据；这段路径不读取支持等级、源轮或真假/否定证据门槛。键本身包含subject与pref_type，因此不是仅凭共同作品就直接合并不同人的偏好。

这条路径能增加已有Memory的曝光，不能从没有源邻轮的泛偏好里凭空恢复吉娜“我也是”或内特最爱提问限定。若要依图结构过滤来源强度，需要先定义并保存字段，再明确检索如何消费；仅在提取提示里要求source_turn_ids，对当前S1归一化和构图不够。

## 结论、置信度、下一步

高置信：缓存ID覆盖、选定缓存原文、真实代码归一化/本地回放节点边及字段消费。合成探针仅证明结构接口，不证明总体图损害或来源规则收益。真实图服务状态、缺缓存的自动重建结果与中文答案净收益仍未知。

1. 增加来源字段时贯通Memory事实侧车、结构归一化、schema、构图和检索门控；保留到原Memory的连接及源轮，勿仅改提示。先离线定义支持等级与时间状态，避免扩schema后仍未消费。
2. 当前中文构图先检查所有输入Memory ID的缓存覆盖与内容/提取版本，缺项须明确提取或拒绝静默当空；不能复用旧conv42缓存冒充当前图。
3. 节点可按主体/对象聚合，但不同支持强度、时间和来源应保留在事实记录或关联边，查询时筛选。方案尚需离线兼容/回放，不能立即当收益证明。
4. 下一阶段核RSI或构图入口是否已检查缓存完整性及提取失败；据入口行为判断缺缓存能否静默造成派生信息减少。服务受限时继续用代码与离线模拟推进，不重打429。
