# 多图探索实验台账（append-only，防重复提案）

> WikiSkill 式审计：每轮/每 wave 的关键决策与被否决方案都记录在案，
> 后续轮次先读此文件，禁止重复被否决的方向。

## 评测方法学决定（最高优先级，先于任何图设计）

1. **dev/test 按 conversation 切分**：dev=7 对话 / test=3 对话（`data/search/split.json`）。
   100 轮的选择信号只来自 dev；test 只用于最终获胜者确认。
2. **同条件参考臂**（2026-09-24 发现并修复）：全量 run 在高并发重压下产生更多退化
   答案（CoT 泄漏、冗长），直接拿它当参考会给所有轮次 +20 左右的系统性假增益
   （r000 首次校准 Δ=+23，4.7σ 不对称）。修复：参考臂 = 校准轮 r000 的逐题结果
   （`--ref-round`），轮次与参考同条件。**第二次校准 Δ=+4（↑44 ↓40）确认真噪声带
   约 ±6~9，|Δ|≥12 才认为可分辨。**
3. **两级判分**：内环 glm-5.3-flash 严格判分（与 5.3 一致率 82.8%，分歧集中在
   flash 偏宽的 exact/partial 边界；配对差值下系统性偏差抵消）；
   最终获胜者用 glm-5.3 非 flash @≤2 并发终审 + test 对话确认。
4. **选择信号 = 执行层**（RSI 综述）：对金标的严格判分，绝不用生成模型自评。
5. 最终多配置选择将用 bootstrap 稳定性（ESPO，B=20）而非点估计。

## 图变体清单（已建，全部幂等，markers 在 data/graph/variants/）

| id | 内容 | 规模 | 成本 |
|---|---|---|---|
| s1 | Entity/Event/Preference + 关系边 | 1455/1074/978 顶点 | 已有 |
| s2 | s1 + 固定词表主题 Concept | +274 顶点/3787 边 | 1 次 LLM |
| v4session | 会话聚合节点 | 272 顶点/1829 边 | 0 |
| v5month | 月份时间片节点 | 269 顶点/1829 边 | 0 |
| v6pair | 实体共现对节点 | 2706 顶点/4094 边 | 0 |
| v8cluster | 嵌入 k-means 聚类节点 | 68 顶点/1829 边 | 1 次嵌入 |
| v10concept | AutoSchemaKG 式实体多级概念化 | 2794 顶点/12075 边 | 1 次 LLM（1035/1455 实体，可回填） |

## Wave 1（14 轮，运行中）：校准 + answer 轴 + 单变体 + 参数

- r000_calib：同条件参考臂，exact 327/616（53.1%）
- r001_precise（answer 轴 official_precise）：**Δ=+17（344），↑58 ↓41，超噪声带——真实提升**；
  partial 175→142（精准输出要求把"模糊对"转成"精确对"）

## Wave 1 结果（2026-09-24，15 轮，dev 616 题，同条件参考 327）

漂移控制：r000_calib(T0)=327 → r014_recalib(Tend)=333，全程漂移 +6；噪声半径 ±6，
|Δ|≥12 判真。按 exact 排名（前五）：

| round | exact | 配置 | 判定 |
|---|---|---|---|
| r012_pernode30 | 347 | s2, per_node=30 | 真效应（+20 vs 校准均值 330） |
| r013_s2v8_precise | 345 | s2+v8cluster, precise | 真效应 |
| r001_precise | 344 | s2, precise | 真效应（**answer 轴最大单项 +17**） |
| r009_all_precise | 342 | 5 图全开, precise | 真效应 |
| r002_base | 339 | 无图 | 高于 s2 校准——dev 集选择偏差（fused 失败富集，任何异配置都有新鲜机会）+ 漂移 |

分题型：precise 轴集中在 temporal（57→64-68）与 multi-hop（45→54-55）；
single-hop 轻微受损（211→204-207）。

**Wave 1 结论**：
1. answer 轴（精准输出要求）是最大、最稳定的真实增益；与图组合不冲突（r013/r009）
2. per_node 15→30（聚合更宽）有效 +20
3. 确定性图变体（v4/v5/v6/v8）单独与全开（r008_all 328）都无增益——
   **"多图量变引起质变"在朴素聚合召回下未成立**，需要更聪明的机制（wave 2 的
   概念图 + wave 3 的反馈边权/路由）
4. seed_k=8（-4）变差：更多种子引入噪声候选
5. bootstrap（B=20）：r013 胜 9 次 / r012 胜 6 次——top2 不稳定，需终审

## Wave 2 结果（12 轮，v10 概念化 + 交叉）

| round | exact | 配置 | 判定 |
|---|---|---|---|
| **r111_big_pn30_precise** | **350（Δ+23）** | s2+v8cluster+v10concept, per_node=30, precise | **全场最佳** |
| r012_pernode30 | 347 | s2, pn30, official | 复用了 wave1 |
| r013_s2v8_precise | 345 | s2+v8, precise | |
| r001_precise | 344 | s2, precise | |
| r110_pn30_precise | 326 | s2, pn30, precise | **交叉反例**：s2 上 pn30×precise 不加 |
| r100_v10 单独 | 325 | v10concept, official | 概念图单独无增益 |
| r104_big（6图全开，official） | 324 | | 全家桶不加 precise 更差 |

Wave 2 结论：
1. **r111（图三组合 s2+v8+v10 × pn30 × precise）= 350 是唯一显著超越单效应叠加点的轮**，
   但 r110（s2 上同样交叉）失败 → 组合增益依赖 v8/v10 与宽聚合的交互，不稳定待复现
2. v10 概念图（AutoSchemaKG 式）单独无增益、与 s2 组合（r102 +4）也不显著——
   论文的"概念边最大增益"在朴素聚合+rerank 融合下**未复现**
3. 分题型：r111 的 temporal 69 是全场最高（precise 轴 + 图时间聚合协同）

## 中断（2026-09-24 06:5x）：DashScope 账户欠费

`Arrearage`（阿里云百炼欠费）→ embedding 与 rerank 全部 400，检索层不可用；
BigModel（LLM/判题）正常。wave 3（6 轮：复现 + pn50/pool45/+v5/消融）配置就绪
（`config/search_rounds_w3.yaml`），充值后：

```bash
.venv/bin/python -u scripts/search_rounds.py --rounds config/search_rounds_w3.yaml
```

最终确认协议（wave 3 后执行）：
1. 幸存配置在 test 对话（data/search/split.json 的 test_convs，从未参与选择）
2. 全量 1986：
   `.venv/bin/python -u scripts/eval_full_fused.py --eval-only --graphs s2,v8cluster,v10concept --per-node 30 --preset official_precise --tag winner`
3. glm-5.3 终审 @2 并发：
   `.venv/bin/python -u scripts/regrade_precision.py winner=results/full_locomo_winner_*.jsonl --pair base winner`
4. 对 base/bypass/fused 三臂配对 + bootstrap 选择

## 待验证/待否决（wave 3 候选）

- SE-GoS 式反馈边权：轮次 via 归因 → 节点效用 → 入池排序（需先给轮次行加 via 记录）
- per_node=50 / pool=45 继续扫
- answer 轴 v2：列表先行格式；adversarial abstention（J 外，低优先）
- 最终获胜者：answer_samples=3 多数票 + test 对话确认
- 被否决：无图单独增益解读（dev 选择偏差，见上）；seed_k=8

## 基础事实（对照用）

- fused 全量（参考条件外）：宽松 J 92.9% / 5.3 严格 exact 67.6%
- dev 集 616 题 = fused 非 exact（dev 对话）全量 + exact 40% 守卫

## 自动追加（2026-09-24 18:29，27 轮累计）

```
校准轮 exact: [323, 333] → 噪声半径 ≈ ±5

round                exact  Δref   ↑   ↓ partial wrong  graphs/preset
r111_big_pn30_precise   350   +23  56  33     153   113  s2,v8cluster,v10concept | official_precise
r012_pernode30         347   +20  53  33     162   107  s2 | official
r013_s2v8_precise      345   +18  51  33     150   121  s2,v8cluster | official_precise
r001_precise           344   +17  58  41     142   130  s2 | official_precise
r009_all_precise       342   +15  55  40     154   120  s1,v4session,v5month,v6pair,v8cluster | official_precise
r002_base              339   +12  50  38     164   113  无图 | official
r107_time_precise      339   +12  51  39     153   124  s1,v5month,v10concept | official_precise
r010_pool60            337   +10  50  40     163   116  s2 | official
r105_big_precise       337   +10  51  41     157   122  s1,v4session,v5month,v6pair,v8cluster,v10concept | official_precise
r004_v4sess            336    +9  48  39     164   116  s1,v4session | official
r003_s1                335    +8  38  30     172   109  s1 | official
r106_time              335    +8  43  35     170   111  s1,v5month,v10concept | official
r108_pair10            335    +8  45  37     172   109  s1,v6pair,v10concept | official
r014_recalib           333    +6  44  38     170   113  s2 | official
r005_v5month           332    +5  44  39     168   116  s1,v5month | official
r006_v6pair            331    +4  42  38     173   112  s1,v6pair | official
r102_v10s2             331    +4  45  41     170   115  s2,v10concept | official
r103_v10_precise       331    +4  45  41     162   123  v10concept,s1 | official_precise
r109_s2v10_precise     330    +3  46  43     164   122  s2,v10concept | official_precise
r007_v8clu             328    +2  43  41     178   109  s1,v8cluster | official
r008_all               328    +1  38  37     172   116  s1,v4session,v5month,v6pair,v8cluster | official
r101_v10s1             326    -1  44  45     176   114  s1,v10concept | official
r110_pn30_precise      326    -1  48  49     169   121  s2 | official_precise
r100_v10               325    -2  39  41     169   122  v10concept | official
r104_big               324    -3  45  48     182   110  s1,v4session,v5month,v6pair,v8cluster,v10concept | official
r000_calib             323    +4  44  40     175   114  s2 | official
r011_seed8             323    -4  39  43     170   123  s2 | official

分题型 exact：
round                  multi-hop    temporal  single-hop open-domain
r111_big_pn30_precise          51          69         209          21
r012_pernode30                54          58         213          22
r013_s2v8_precise             55          65         205          20
r001_precise                  55          64         207          18
r009_all_precise              55          68         202          17
r002_base                     55          54         211          19
r107_time_precise             47          62         209          21
r010_pool60                   55          54         205          23
r105_big_precise              47          65         205          20
r004_v4sess                   51          57         209          19
r003_s1                       49          57         210          19
r106_time                     53          53         208          21
r108_pair10                   48          60         206          21
r014_recalib                  48          57         207          21
r005_v5month                  46          52         214          20
r006_v6pair                   49          50         216          16
r102_v10s2                    49          52         210          20
r103_v10_precise              47          58         207          19
r109_s2v10_precise            48          65         197          20
r007_v8clu                    48          53         210          17
r008_all                      48          53         209          18
r101_v10s1                    47          53         207          19
r110_pn30_precise             42          62         204          18
r100_v10                      49          54         205          17
r104_big                      46          54         206          18
r000_calib                    45          57         204          17
r011_seed8                    45          51         207          20
```

## 自动追加（2026-09-24 22:32，39 轮累计）

```
校准轮 exact: [323, 333, 335] → 噪声半径 ≈ ±6

round                exact  Δref   ↑   ↓ partial wrong  graphs/preset
r111_big_pn30_precise   350   +23  56  33     153   113  s2,v8cluster,v10concept | official_precise
r012_pernode30         347   +20  53  33     162   107  s2 | official
r013_s2v8_precise      345   +18  51  33     150   121  s2,v8cluster | official_precise
r001_precise           344   +17  58  41     142   130  s2 | official_precise
r411_laya_filter_pn30   344   +17  56  39     153   119  s2 | official_precise
r009_all_precise       342   +15  55  40     154   120  s1,v4session,v5month,v6pair,v8cluster | official_precise
r404_laya_both         340   +13  42  29     156   120  s2 | official
r002_base              339   +12  50  38     164   113  无图 | official
r107_time_precise      339   +12  51  39     153   124  s1,v5month,v10concept | official_precise
r401_r111_rep          338   +11  51  40     152   126  s2,v8cluster,v10concept | official_precise
r010_pool60            337   +10  50  40     163   116  s2 | official
r105_big_precise       337   +10  51  41     157   122  s1,v4session,v5month,v6pair,v8cluster,v10concept | official_precise
r004_v4sess            336    +9  48  39     164   116  s1,v4session | official
r407_pn30_rep          336    +9  47  38     169   111  s2 | official
r003_s1                335    +8  38  30     172   109  s1 | official
r106_time              335    +8  43  35     170   111  s1,v5month,v10concept | official
r108_pair10            335    +8  45  37     172   109  s1,v6pair,v10concept | official
r400_calib3            335    +8  45  37     170   111  s2 | official
r408_laya_only_list    334    +7  44  37     173   109  s2 | official
r014_recalib           333    +6  44  38     170   113  s2 | official
r005_v5month           332    +5  44  39     168   116  s1,v5month | official
r405_laya_r111         332    +5  44  39     153   131  s2,v8cluster,v10concept | official_precise
r006_v6pair            331    +4  42  38     173   112  s1,v6pair | official
r102_v10s2             331    +4  45  41     170   115  s2,v10concept | official
r103_v10_precise       331    +4  45  41     162   123  v10concept,s1 | official_precise
r402_laya_route        331    +4  41  37     170   115  s2 | official
r410_seed3             331    +4  43  39     175   110  s2 | official
r109_s2v10_precise     330    +3  46  43     164   122  s2,v10concept | official_precise
r403_laya_filter       330    +3  47  44     165   120  s2 | official
r007_v8clu             328    +2  43  41     178   109  s1,v8cluster | official
r008_all               328    +1  38  37     172   116  s1,v4session,v5month,v6pair,v8cluster | official
r101_v10s1             326    -1  44  45     176   114  s1,v10concept | official
r110_pn30_precise      326    -1  48  49     169   121  s2 | official_precise
r100_v10               325    -2  39  41     169   122  v10concept | official
r104_big               324    -3  45  48     182   110  s1,v4session,v5month,v6pair,v8cluster,v10concept | official
r000_calib             323    +4  44  40     175   114  s2 | official
r011_seed8             323    -4  39  43     170   123  s2 | official
r406_laya_precise      320    -7  37  44     172   124  s2 | official_precise
r409_norerank          313   -14  44  58     173   130  s2 | official

分题型 exact：
round                  multi-hop    temporal  single-hop open-domain
r111_big_pn30_precise          51          69         209          21
r012_pernode30                54          58         213          22
r013_s2v8_precise             55          65         205          20
r001_precise                  55          64         207          18
r411_laya_filter_pn30          51          64         209          20
r009_all_precise              55          68         202          17
r404_laya_both                49          60         210          21
r002_base                     55          54         211          19
r107_time_precise             47          62         209          21
r401_r111_rep                 48          67         201          22
r010_pool60                   55          54         205          23
r105_big_precise              47          65         205          20
r004_v4sess                   51          57         209          19
r407_pn30_rep                 46          57         212          21
r003_s1                       49          57         210          19
r106_time                     53          53         208          21
r108_pair10                   48          60         206          21
r400_calib3                   50          57         208          20
r408_laya_only_list           51          57         210          16
r014_recalib                  48          57         207          21
r005_v5month                  46          52         214          20
r405_laya_r111                50          63         198          21
r006_v6pair                   49          50         216          16
r102_v10s2                    49          52         210          20
r103_v10_precise              47          58         207          19
r402_laya_route               48          54         210          19
r410_seed3                    52          52         210          17
r109_s2v10_precise            48          65         197          20
r403_laya_filter              54          50         203          23
r007_v8clu                    48          53         210          17
r008_all                      48          53         209          18
r101_v10s1                    47          53         207          19
r110_pn30_precise             42          62         204          18
r100_v10                      49          54         205          17
r104_big                      46          54         206          18
r000_calib                    45          57         204          17
r011_seed8                    45          51         207          20
r406_laya_precise             44          58         197          21
r409_norerank                 43          50         203          17
```

## 自动追加（2026-09-25 03:25，50 轮累计）

```
校准轮 exact: [323, 333, 335, 348] → 噪声半径 ≈ ±12

round                exact  Δref   ↑   ↓ partial wrong  graphs/preset
r111_big_pn30_precise   350   +23  56  33     153   113  s2,v8cluster,v10concept | official_precise
r501_calib4            348   +21  50  29     154   114  s2 | official
r012_pernode30         347   +20  53  33     162   107  s2 | official
r507_pool45_precise    346   +19  51  32     150   120  s2 | official_precise
r013_s2v8_precise      345   +18  51  33     150   121  s2,v8cluster | official_precise
r001_precise           344   +17  58  41     142   130  s2 | official_precise
r411_laya_filter_pn30   344   +17  56  39     153   119  s2 | official_precise
r500_s3_precise        343   +16  51  35     146   127  s2 | official_precise
r502_s3_nograph_precise   343   +16  54  38     158   115  无图 | official_precise
r009_all_precise       342   +15  55  40     154   120  s1,v4session,v5month,v6pair,v8cluster | official_precise
r503_s2v8_precise_rep   342   +15  51  36     149   125  s2,v8cluster | official_precise
r509_seed3_precise     341   +14  51  37     161   114  s2 | official_precise
r404_laya_both         340   +13  42  29     156   120  s2 | official
r505_rerank_off_precise   340   +13  68  55     145   131  s2 | official_precise
r508_s1_precise        340   +13  53  40     151   125  s1 | official_precise
r002_base              339   +12  50  38     164   113  无图 | official
r107_time_precise      339   +12  51  39     153   124  s1,v5month,v10concept | official_precise
r504_laya_both_rep     339   +12  47  35     165   112  s2 | official
r401_r111_rep          338   +11  51  40     152   126  s2,v8cluster,v10concept | official_precise
r506_s3_precise_rep    338   +11  49  38     158   120  s2 | official_precise
r010_pool60            337   +10  50  40     163   116  s2 | official
r105_big_precise       337   +10  51  41     157   122  s1,v4session,v5month,v6pair,v8cluster,v10concept | official_precise
r004_v4sess            336    +9  48  39     164   116  s1,v4session | official
r407_pn30_rep          336    +9  47  38     169   111  s2 | official
r003_s1                335    +8  38  30     172   109  s1 | official
r106_time              335    +8  43  35     170   111  s1,v5month,v10concept | official
r108_pair10            335    +8  45  37     172   109  s1,v6pair,v10concept | official
r400_calib3            335    +8  45  37     170   111  s2 | official
r408_laya_only_list    334    +7  44  37     173   109  s2 | official
r014_recalib           333    +6  44  38     170   113  s2 | official
r005_v5month           332    +5  44  39     168   116  s1,v5month | official
r405_laya_r111         332    +5  44  39     153   131  s2,v8cluster,v10concept | official_precise
r006_v6pair            331    +4  42  38     173   112  s1,v6pair | official
r102_v10s2             331    +4  45  41     170   115  s2,v10concept | official
r103_v10_precise       331    +4  45  41     162   123  v10concept,s1 | official_precise
r402_laya_route        331    +4  41  37     170   115  s2 | official
r410_seed3             331    +4  43  39     175   110  s2 | official
r109_s2v10_precise     330    +3  46  43     164   122  s2,v10concept | official_precise
r403_laya_filter       330    +3  47  44     165   120  s2 | official
r007_v8clu             328    +2  43  41     178   109  s1,v8cluster | official
r008_all               328    +1  38  37     172   116  s1,v4session,v5month,v6pair,v8cluster | official
r510_allv8_precise     328    +1  43  42     160   128  s1,v4session,v5month,v6pair,v8cluster | official_precise
r101_v10s1             326    -1  44  45     176   114  s1,v10concept | official
r110_pn30_precise      326    -1  48  49     169   121  s2 | official_precise
r100_v10               325    -2  39  41     169   122  v10concept | official
r104_big               324    -3  45  48     182   110  s1,v4session,v5month,v6pair,v8cluster,v10concept | official
r000_calib             323    +4  44  40     175   114  s2 | official
r011_seed8             323    -4  39  43     170   123  s2 | official
r406_laya_precise      320    -7  37  44     172   124  s2 | official_precise
r409_norerank          313   -14  44  58     173   130  s2 | official

分题型 exact：
round                  multi-hop    temporal  single-hop open-domain
r111_big_pn30_precise          51          69         209          21
r501_calib4                   53          59         214          22
r012_pernode30                54          58         213          22
r507_pool45_precise           49          63         213          21
r013_s2v8_precise             55          65         205          20
r001_precise                  55          64         207          18
r411_laya_filter_pn30          51          64         209          20
r500_s3_precise               54          62         211          16
r502_s3_nograph_precise          47          69         204          23
r009_all_precise              55          68         202          17
r503_s2v8_precise_rep          52          63         207          20
r509_seed3_precise            54          65         204          18
r404_laya_both                49          60         210          21
r505_rerank_off_precise          53          62         203          22
r508_s1_precise               52          66         205          17
r002_base                     55          54         211          19
r107_time_precise             47          62         209          21
r504_laya_both_rep            52          60         209          18
r401_r111_rep                 48          67         201          22
r506_s3_precise_rep           53          66         201          18
r010_pool60                   55          54         205          23
r105_big_precise              47          65         205          20
r004_v4sess                   51          57         209          19
r407_pn30_rep                 46          57         212          21
r003_s1                       49          57         210          19
r106_time                     53          53         208          21
r108_pair10                   48          60         206          21
r400_calib3                   50          57         208          20
r408_laya_only_list           51          57         210          16
r014_recalib                  48          57         207          21
r005_v5month                  46          52         214          20
r405_laya_r111                50          63         198          21
r006_v6pair                   49          50         216          16
r102_v10s2                    49          52         210          20
r103_v10_precise              47          58         207          19
r402_laya_route               48          54         210          19
r410_seed3                    52          52         210          17
r109_s2v10_precise            48          65         197          20
r403_laya_filter              54          50         203          23
r007_v8clu                    48          53         210          17
r008_all                      48          53         209          18
r510_allv8_precise            49          62         201          16
r101_v10s1                    47          53         207          19
r110_pn30_precise             42          62         204          18
r100_v10                      49          54         205          17
r104_big                      46          54         206          18
r000_calib                    45          57         204          17
r011_seed8                    45          51         207          20
r406_laya_precise             44          58         197          21
r409_norerank                 43          50         203          17
```

## 终审：同夜背靠背双臂全量（2026-09-25 02:36，glm-5.3 @2 并发严格判分）

设计：A=无图基线（official）与 B=全栈（s2+v8 融合、pn30、precise）同一夜晚背靠背
各跑全量 1986，消除夜间漂移；test 对话从未参与 50 轮中任何选择。

| 口径 | A 基线 | B 全栈 | 配对 |
|---|---|---|---|
| 宽松 J | 92.7% | 92.0% | — |
| **精准 exact** | **66.4%（1022/1539）** | **66.8%（1029/1540）** | Δ+7，McNemar p=0.64 **无差异** |
| temporal | 54.1% | 57.3% | +3.2pp（唯一方向性优势，被噪声覆盖） |
| adversarial 干净拒答 | 70 | 61 | 略差，n.s. |
| dev 对话 Δ | — | +6 | test 对话 Δ=+1 ← **dev 增益在 test 上消失（选择偏差确认）** |

## 50 轮最终结论（wave 1-6，2026-09-24~25）

1. **精准口径 66.4-66.8% 是当前栈（mem0 提取 + 向量检索 + rerank）的硬区间**；
   剩余 33pp 的构成：提取层丢失（single-hop 76.7% 封顶）> 列表缺项 > 同义判 partial 长尾
2. **所有检索侧/answer 侧配置轴在控变量后全部落入噪声带**：图开关（Δ+7 n.s.）、
   多图组合、聚合宽度、候选池、种子数、采样数、Laya 路由/过滤——无一幸存复现
3. **此前报告的正效应全部是伪影**：precise +17 / pn30 +20 / r111 +23 复现轮全部失败
   （r401=338 vs 原 350；r407=336 vs 原 347；同夜 official 校准一度到 348 超过所有 precise 轮）
4. 唯一可靠效应：**rerank 关闭 -22（两次验证）——rerank 必须保留**
5. 夜间条件漂移（±15-20 题）> 任何配置效应（±5-8 题）；两臂间约 15% 的题答案
   完全不同但净值≈0（167 次双向翻转，80 vs 87）
6. 方法学资产被验证：同夜括号校准 + 复现闸门 + test 对话保留，正是抓住上述伪影的原因

## 下一步唯一有希望的方向（按天花板排序）

1. **提取层重做**（single-hop 22% 缺口全在这）：两轮提取/证据链保留，需重灌 + 全量重跑
2. 列表完整性：检索已带全（rerank 后仍缺项说明是提取端没存全）
3. 判分长尾：同义改写判 partial 属于精准口径的固有代价

## 附验二：提取 vs 召回四级硬拆（2026-09-25，A 臂 513 非 exact 非 adversarial 题）

工具：`scripts/verify_extraction_tier.py split`（词边界匹配 + 证据 session 约束：证据 session
记忆 ≥半数词命中算已存，跨 session 记忆须全词命中）。610 条缺失事实按管线顺序互斥归因：

| 桶 | 条目(原始) | 条目(校准*) | 涉及题目 | 题目主归因 |
|---|---|---|---|---|
| 提取丢失（对话明说、库没有） | 132 | **~90** | 123 | 123 |
| 召回丢失（库有、没进 top-15 上下文） | 74 | **~55** | 63 | 59 |
| 上下文已有（进了上下文仍没答出） | 202 | 下界 | 167 | 147 |
| 对话没明说（temporal 日期计算 114 为主） | 202 | — | 198 | 184 |

*校准：分层抽样 20 条人工判读（提取 12 + 召回 8），提取桶精度 ~70%（判分元词泄漏虚增、
session 约束过严把已存事实误入各占一半，如 conv-41_qa45 的 military+running for office
其实都在库里），召回桶 ~75%（单词条目撞进无关 session 记忆，如 "traveling"）。两个方向
的误差部分对冲，提取真实值与附验一第一级口径的 88 条互相印证。

**提取丢失的真实构成**：推荐清单成员（"Project Hail Mary"）、引语/感受词（"super cozy"、
"magical"、"ongoing adventure of learning and growing"）、话题总结类——与附验一"官方
prompt 收录范围"的结论一致；**召回丢失的真实构成**：城市/地名（Seattle 连丢 3 题）、书名
（Sapiens）、限量款服饰——多为列表题的单个成员被 top-15 挤掉，或同义改写
（yellow mug ↔ coffee cup）。注意 chroma 无 BM25（启动即警告 hybrid 被禁用），纯语义
检索对专名/引语类弱——召回桶的可行修法是混合检索（qdrant/es 的 keyword search）或
查询改写，而非继续调配置轴（50 轮已证配置轴全噪声）。

分数影响上界（每桶全修好时最多可翻的题数）：提取 ~123、召回 ~59、上下文已有 ~147、
没明说 ~184（多数不可修，temporal 日期计算属作答侧推理）。

## 附验一：换非 flash 提取能否补漏（2026-09-25，conv-42/43 隔离重灌）

问题：提取层丢失是否因 `LLM_MODEL=glm-5.3-flash` 档位不够？换强模型是不是廉价捷径？

方法：conv-42/43 用 glm-5.3 非 flash 完整重灌（user 前缀 `nf53:` 隔离，embedder/prompt/
管线与 full run 完全一致；conv-43 session_17 曾 429 跳过、已补灌）。对照清单来自终审 A 臂
非 exact 题的"缺失事实 → 库内存在性"归因（scripts/verify_extraction_tier.py：judge.missing/gold
内容词过滤判分抱怨类元词 + 要求词组在证据原文共现）。

**归因口径（两级复核后）**：517 非 exact 非 adversarial 题的 614 条缺失事实，第一级按
证据 turn 共现分桶后，对"不可归因"桶再做第二级复核（对照**整段对话原文** + flash 库）：

| 终局归因 | 条数 | 占比 | 说明 |
|---|---|---|---|
| 库里已有（丢在作答/判分） | 334 | 54% | 241 证据词共现命中 + 93 整对话复核捞出 |
| 对话里没明说 | 188 | 31% | temporal 日期/时长计算 108 + 综合推理/超词面转述 80 |
| **硬提取丢失（对话明说、库没有）** | **88** | **14%** | 涉及 86 道题；single-hop 46、open-domain 17 |

88 条中评价/引语类为主（"rejections don't define her"、"brave, selfless"），也含少量
硬实体（书名 "A Court of Thorns and Roses"、"Trans Lives Matter"）。

结果：conv-42/43 的全部 26 条丢失条目对非 flash 重灌库复查——**仅补回 3 条（12%）**；
记忆总量 flash 192/207 vs nf 202/212（+3~5%）。抽查仍缺项：相邻的具体事件都已入库
（rejection 事件、The Alchemist 重读、game tournament），被丢的只有引语/评价——
两档模型行为一致（补回的恰是 3 条硬实体类，进一步佐证档位只影响边角）。

结论：**提取丢失的根因是 mem0 官方 prompt 的收录范围（7 类个人硬信息 + 具名 few-shot），
与模型档位无关；"换强模型"捷径无效。** 提取层重做（方向 1）必须改 prompt/两轮提取；
完美提取的天花板 = 86 道题 ≈ 精准 +5.6pp（66.4%→≤72%），而非整段 33pp——
且大头（54%）其实在作答/判分侧，与方向 3 重叠。

## 自动追加（2026-09-25 09:05，58 轮累计）

```
校准轮 exact: [323, 333, 335, 348] → 噪声半径 ≈ ±12

round                exact  Δref   ↑   ↓ partial wrong  graphs/preset
r607_all               380   +53  95  42     139    97  s2 | official_precise
r605_jev_stop          379   +52  88  36     146    91  s2 | official
r601_atomic            373   +46  79  33     155    88  s2 | official
r603_atomic_precise    370   +43  82  39     142   104  s2 | official_precise
r604_atomic_nograph    367   +40  75  35     152    97  无图 | official
r606_jev_filter        364   +37  73  36     159    93  s2 | official
r602_atomic_rep        357   +30  79  49     163    95  s2 | official
r111_big_pn30_precise   350   +23  56  33     153   113  s2,v8cluster,v10concept | official_precise
r501_calib4            348   +21  50  29     154   114  s2 | official
r012_pernode30         347   +20  53  33     162   107  s2 | official
r600_noatomic          347   +20  51  31     152   117  s2 | official
r507_pool45_precise    346   +19  51  32     150   120  s2 | official_precise
r013_s2v8_precise      345   +18  51  33     150   121  s2,v8cluster | official_precise
r001_precise           344   +17  58  41     142   130  s2 | official_precise
r411_laya_filter_pn30   344   +17  56  39     153   119  s2 | official_precise
r500_s3_precise        343   +16  51  35     146   127  s2 | official_precise
r502_s3_nograph_precise   343   +16  54  38     158   115  无图 | official_precise
r009_all_precise       342   +15  55  40     154   120  s1,v4session,v5month,v6pair,v8cluster | official_precise
r503_s2v8_precise_rep   342   +15  51  36     149   125  s2,v8cluster | official_precise
r509_seed3_precise     341   +14  51  37     161   114  s2 | official_precise
r404_laya_both         340   +13  42  29     156   120  s2 | official
r505_rerank_off_precise   340   +13  68  55     145   131  s2 | official_precise
r508_s1_precise        340   +13  53  40     151   125  s1 | official_precise
r002_base              339   +12  50  38     164   113  无图 | official
r107_time_precise      339   +12  51  39     153   124  s1,v5month,v10concept | official_precise
r504_laya_both_rep     339   +12  47  35     165   112  s2 | official
r401_r111_rep          338   +11  51  40     152   126  s2,v8cluster,v10concept | official_precise
r506_s3_precise_rep    338   +11  49  38     158   120  s2 | official_precise
r010_pool60            337   +10  50  40     163   116  s2 | official
r105_big_precise       337   +10  51  41     157   122  s1,v4session,v5month,v6pair,v8cluster,v10concept | official_precise
r004_v4sess            336    +9  48  39     164   116  s1,v4session | official
r407_pn30_rep          336    +9  47  38     169   111  s2 | official
r003_s1                335    +8  38  30     172   109  s1 | official
r106_time              335    +8  43  35     170   111  s1,v5month,v10concept | official
r108_pair10            335    +8  45  37     172   109  s1,v6pair,v10concept | official
r400_calib3            335    +8  45  37     170   111  s2 | official
r408_laya_only_list    334    +7  44  37     173   109  s2 | official
r014_recalib           333    +6  44  38     170   113  s2 | official
r005_v5month           332    +5  44  39     168   116  s1,v5month | official
r405_laya_r111         332    +5  44  39     153   131  s2,v8cluster,v10concept | official_precise
r006_v6pair            331    +4  42  38     173   112  s1,v6pair | official
r102_v10s2             331    +4  45  41     170   115  s2,v10concept | official
r103_v10_precise       331    +4  45  41     162   123  v10concept,s1 | official_precise
r402_laya_route        331    +4  41  37     170   115  s2 | official
r410_seed3             331    +4  43  39     175   110  s2 | official
r109_s2v10_precise     330    +3  46  43     164   122  s2,v10concept | official_precise
r403_laya_filter       330    +3  47  44     165   120  s2 | official
r007_v8clu             328    +2  43  41     178   109  s1,v8cluster | official
r008_all               328    +1  38  37     172   116  s1,v4session,v5month,v6pair,v8cluster | official
r510_allv8_precise     328    +1  43  42     160   128  s1,v4session,v5month,v6pair,v8cluster | official_precise
r101_v10s1             326    -1  44  45     176   114  s1,v10concept | official
r110_pn30_precise      326    -1  48  49     169   121  s2 | official_precise
r100_v10               325    -2  39  41     169   122  v10concept | official
r104_big               324    -3  45  48     182   110  s1,v4session,v5month,v6pair,v8cluster,v10concept | official
r000_calib             323    +4  44  40     175   114  s2 | official
r011_seed8             323    -4  39  43     170   123  s2 | official
r406_laya_precise      320    -7  37  44     172   124  s2 | official_precise
r409_norerank          313   -14  44  58     173   130  s2 | official

分题型 exact：
round                  multi-hop    temporal  single-hop open-domain
r607_all                      59          82         219          20
r605_jev_stop                 59          75         227          18
r601_atomic                   58          72         222          21
r603_atomic_precise           52          81         217          20
r604_atomic_nograph           58          64         226          19
r606_jev_filter               59          72         216          17
r602_atomic_rep               49          76         212          20
r111_big_pn30_precise          51          69         209          21
r501_calib4                   53          59         214          22
r012_pernode30                54          58         213          22
r600_noatomic                 50          61         216          20
r507_pool45_precise           49          63         213          21
r013_s2v8_precise             55          65         205          20
r001_precise                  55          64         207          18
r411_laya_filter_pn30          51          64         209          20
r500_s3_precise               54          62         211          16
r502_s3_nograph_precise          47          69         204          23
r009_all_precise              55          68         202          17
r503_s2v8_precise_rep          52          63         207          20
r509_seed3_precise            54          65         204          18
r404_laya_both                49          60         210          21
r505_rerank_off_precise          53          62         203          22
r508_s1_precise               52          66         205          17
r002_base                     55          54         211          19
r107_time_precise             47          62         209          21
r504_laya_both_rep            52          60         209          18
r401_r111_rep                 48          67         201          22
r506_s3_precise_rep           53          66         201          18
r010_pool60                   55          54         205          23
r105_big_precise              47          65         205          20
r004_v4sess                   51          57         209          19
r407_pn30_rep                 46          57         212          21
r003_s1                       49          57         210          19
r106_time                     53          53         208          21
r108_pair10                   48          60         206          21
r400_calib3                   50          57         208          20
r408_laya_only_list           51          57         210          16
r014_recalib                  48          57         207          21
r005_v5month                  46          52         214          20
r405_laya_r111                50          63         198          21
r006_v6pair                   49          50         216          16
r102_v10s2                    49          52         210          20
r103_v10_precise              47          58         207          19
r402_laya_route               48          54         210          19
r410_seed3                    52          52         210          17
r109_s2v10_precise            48          65         197          20
r403_laya_filter              54          50         203          23
r007_v8clu                    48          53         210          17
r008_all                      48          53         209          18
r510_allv8_precise            49          62         201          16
r101_v10s1                    47          53         207          19
r110_pn30_precise             42          62         204          18
r100_v10                      49          54         205          17
r104_big                      46          54         206          18
r000_calib                    45          57         204          17
r011_seed8                    45          51         207          20
r406_laya_precise             44          58         197          21
r409_norerank                 43          50         203          17
```

## Wave 7：提取层重做 + Jev 融入（2026-09-25，同夜配对，dev 616）——突破

背景：50 轮终审归因指向提取层；Jev-Mem 调研（arXiv:2609.23986）给出四维评分/停止准则。
本轮两大新基建：**4888 条原子事实**（一事一条+日期锚定，独立向量池 RRF 合流，
mem0 主池不动）+ **Laya/decider 双端点接入**（四维并行 noul、证据充足判定）。

| 轮 | 配置 | exact | 判定 |
|---|---|---|---|
| r600 对照（无原子池） | s2, official | 347 | 同夜对照 |
| r601 原子池 | +atomic | **373** | **提取层效应 +26** |
| r602 原子池复现 | 同上 | 357 | 复现 +10（效应真实，幅度 +10~+26） |
| r603 原子+precise | | 370 | |
| r604 原子+无图 | | 367 | 图仍无独立贡献 |
| r605 原子+Jev停止 | | **379** | 证据不足→加倍聚合重拉，+6~+22 |
| r606 原子+四维过滤 | | 364 | 过滤略负（丢支持性记忆） |
| **r607 全叠加** | 原子+停止+过滤+precise | **380** | **全场历史最高** |

结论：
1. **提取层重做是全场最大真实效应**（同夜配对 + 复现双重验证）：
   原子化提取（一事一条、字面量保留、日期锚定、列表逐项拆）直接补上
   single-hop 缺口与列表缺项
2. **Jev 停止准则有效**（证据不足→扩拉），四维过滤略负——过滤阈值 0.35 偏激进，
   待调
3. dev 380/616（61.7%）vs 同夜对照 347（56.3%）；换算全量约 66.4%→71-72%（未做
   全量终审，用户指示到本轮为止）
4. 方法教训再次验证：这轮所有结论都来自同夜配对 + 复现轮，与 50 轮的教训一致

## 附验三：Laya 提取门控 POC（2026-09-25，conv-42 单对话双服务，工作流跑批）

想法：按 session 提取后让 Laya 决策服务只答三问（missing/unsupported/conflict，choice 二分类
取 P(B)，不输出原因），风险超阈值升级 glm-5.3 二审。工具 `scripts/laya_extraction_poc.py`
（端点自动降级：本机→公网→DoH 绕过本机 fake-IP DNS 劫持），数据 `results/laya_poc/`，
完整报告 `poc_report.md`。

- **检测力**：decider-2b AUC 0.698（按修正标签 0.795，CI 不含 0.5）；laya 完整版 0.569
  无区分力（分数挤在 0.34-0.53 窄带，与内容基本无关）→ **小模型反而适合当门控**
- **阈值**：decider missing P(B)≥0.3 → 升级 3/29 且全真漏；但漏检 79%——高精度低召回的
  兜底门，不是主力质检
- **收益上限**：nf53 对照显示强模型二审只覆盖漏点的 12%~33% → **『风险→glm-5.3 二审』
  链路现阶段不值得接入**；正确顺序 = 先改提取 prompt 收录范围（方向 1），decider@0.3
  留作修复后的低成本复验门控
- **重要副产品（修正附验二口径）**：session 级真值标签有系统性噪声——"Project Hail Mary"
  全对话 0 次出现（QA gold 泄漏进 missing 条目）、"A Court of Thorns and Roses"只在
  session_19 出现却被归因到 5 个 session、且 flash 库里已有该书名记忆；按"漏点关键短语
  须在被归因 session 自己的 turn 内共现"修正后正例 14→8。**后续 session 级归因一律用此口径**
- 环境：本机 6006/6008 当前未监听；公网 decider 是 uu 前缀同域（laya.py:28 为准）

## 最终终审（2026-09-25 上午，同夜双臂全量 + 全量严格判分 judge-all，无宽判预筛）

| 臂 | 配置 | 精准 exact |
|---|---|---|
| C 对照 | 无图无原子池 official | **66.6%**（1026/1540） |
| D 冠军 | 原子池4888 + Jev停止 + 四维过滤 + precise | **70.1%**（1080/1540） |

- **Δ=+54 题，McNemar p=0.0003 —— 全 campaign 首个统计显著的正效应**
- 翻转：C 独有 80 / D 独有 134；dev Δ+41 / test Δ+13（test 上也正，效应真实）
- **temporal 54.5%→68.5%（+14pp）** 是最大受益题型（日期锚定原子事实+精准措辞直接对症）；
  multi-hop +2.1pp；single-hop/open-domain 持平
- 对抗题（judge-all 口径）：C 干净拒答 129 vs D 114（n.s.；precise 措辞略增附猜测）
- 宽松判分按用户指示弃用；全部 1540×2 题经 glm-5.3 @2 并发逐题严格判分
