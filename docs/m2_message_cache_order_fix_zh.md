# Mem0 历史消息缓存顺序修复与旧审计保真

> 2026-09-28。对应此前 [272 场顺序审计](m2_last_messages_tie_zh.md) 与 [有序缓存离线 POC](m2_ordered_message_cache_poc_zh.md)。实现改动在 vendored `third_party/mem0-src/mem0/memory/storage.py`；真实中文重放和旧归档对账由 `scripts/m2_verify_message_cache_order_fix.py` 完成，机器结果为 `results/analysis/m2_message_cache_order_fix_20260928.json`。零模型调用，零真实记忆库、Chroma 或图写入。

## 问题与修正

同一 `save_messages` 批次所有消息共享一个 `created_at`。旧 SQL 在留存和读取时只按时间排序；相同时间戳下，本机 272/272 个中文长批次留下的是**开头 10 条**。这把前场早期原话持续放进下一场的 `Last k Messages`。

现将留存选择改为 `created_at DESC, rowid DESC`，读取时先用相同规则选最近 N 条，再按 `created_at ASC, rowid ASC` 返回，保证最近窗口和对话顺序一致。没有改 `session_scope`、写入消息、提取提示、图或检索配置，也没有迁移线上/现存数据库。

新增三个存储层回归：单批 25 条时保留末尾 10 条并按原顺序返回；两批时间戳完全相同时保留最后插入者；不同 owner 的窗口隔离且跨批时间推进正确。与来源追踪相关测试合跑 **9 passed**。生产 `SQLiteManager` 在隔离内存库上重新喂入中文 LoCoMo 10 个对话、272 场：272/272 个长批次保留末尾 10 条；封存的旧实现仍为 272/272 留开头 10 条。两条已核错挂病例的旧源 turn 都从修复后的 `Last k` 窗口移出，但 `Existing Memories` 通道未受本修复影响。

## 历史结果可复算

三个旧脚本原来直接导入当前 `SQLiteManager`，修复后继续运行会把“当时实现”的审计悄悄变成新行为。新增只允许 `:memory:` 的 `LegacySQLiteManager`，保留修复前的两段 SQL，仅供离线旧结果重放；旧 `storage.py` SHA 随原归档固定保存。

`m2_audit_last_messages_tie.py`、`m2_audit_carryover_cache_eligibility.py` 和 `m2_poc_ordered_message_cache.py` 已切到此历史臂。三份脚本将结果写入临时文件，与各自原 JSON 逐项比较，**三份完全相同**；原归档未覆盖。真实 Mem0 history DB 的运行前后 SHA 也一致。此验证不能代替历史原始模型请求/回复，因为它们本来没有冻结。

## 结论、置信度、下一步优化建议

**结论：** 生产代码现按插入顺序处理同时间戳消息，修复了“最近 10 条”被实际实现为“开头 10 条”的本地可复现错误；历史审计仍保持原版本语义。现存 DB 里已经保留的旧 10 条不会被追溯改写，须待后续新消息写入才更新其窗口。

**置信度：** 对代码行为、三个回归、272 场隔离重放和三份旧归档逐项一致为高。`rowid` 是此 SQLite 表当前的插入顺序代理；如果未来复制、重建或 VACUUM 现存表，要另审这个假设，长期可增加显式序号。对旧场次实际提取依赖哪个输入通道、修复后新记忆质量与答案影响仍未知。

**下一步：** 来源门控要让新 ADD 的每条事实有本场 `New Messages` 的可核 turn；旧缓存与 `Existing Memories` 可辅助解指、归并，却不能变成新事实的来源场次。固定跨场 12 对继续做有源独有细节与无源身份的输出事实验收；答题净收益另做同槽位配对和反伤检查，不从缓存可见性变化直接推出收益。
