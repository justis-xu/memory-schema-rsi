# 当前中文结构缓存：关系来源字段实测

> 2026-09-29。对当前十个 `zhfull:locomo:*` 用户的源 Memory ID 与本地 `structured_cache.json` 直接求交，并检查**原始缓存条目**，不是只看归一化后的 S1 图。脚本 [`m2_audit_cached_relation_provenance.py`](../scripts/m2_audit_cached_relation_provenance.py)与[字段频数、输入指纹](../results/analysis/m2_cached_relation_provenance_20260929.json)留档。零模型调用、零图/记忆写入，源库前后逻辑指纹相同。

## 实测

当前源 1,800 条中，结构缓存按 ID 命中 581 条，缺 1,219 条。581 条的原始缓存共有 **2,532 个结构条目**：实体 1,482、事件 351、偏好 400、关系 299。这是条目数，包含重复描述，不能等同于 2,532 个独有真实事实。

逐条检查 `source_turn_ids`、`source_ref`、`source_channel`、`support_type`、`support_strength`、`source_date`、`dia_id`、`source_session_id` 八种显式来源字段：**2,532 个条目中 0 个带这些字段；581 条命中记忆中 0 条在顶层或条目内带这些字段**。原始条目的可见键仅为现有 S1 语义字段，如实体 `name/type`、事件 `name/date/participants`、偏好 `subject/pref_type/object`、关系 `from_entity/relation/to_entity`。结果并非 `_normalize` 删除后才造成的统计假象；原始缓存本身就没有所检查的来源信息。

已核代码仍会把 Memory ID 连到派生节点；因此不能说图完全没有任何溯源路径。但 Memory ID 只回到压缩记忆，并不证明派生关系由哪个原始 turn、说话者或图片渠道支持。已见过记忆写入场次与事实源场不一致、相邻问答共同决定关系、图片 query/caption 与本人文字支持强度不同。用这 581 条缓存直接构图，即使节点数与源 ID 对齐，也不能通过事实级来源验收。

## 结论、置信度、下一步优化建议

**结论：** 当前命中的结构缓存没有显式事实来源字段；连同 1,219 条缺项，它不能直接成为“完整且有源的当前中文图”输入。源 Memory 的完整枚举已由[上一阶段](m2_current_source_enumeration_zh.md)证明，这里的阻断点是**派生关系的来源与提取版本**，不是源清单分页。

**置信度：** 当前 ID/缓存交集、2,532 条目类型和列举字段的零命中为高。字段缺失不等于每个条目事实错误；可能另有人工来源侧车，但不在这份缓存和现有 S1 构图路径中。缓存缺正文、模型、提示指纹，命中也不能证明仍匹配当前 Memory 内容。

**下一步优化建议：** 不把旧缓存的命中标成已验证关系。小范围按原始 turn 重新提取时，将多轮来源、说话者、渠道和支持强度保存在事实侧车，再验 `extractor → normalize → graph → retrieval` 各步是否保留并使用；只有完整源/提取状态/图版本和被挤出事实可复核，才进入中文同槽答题对照。全量补提 1,219 条超出小 POC，需另定额度和版本冻结。
