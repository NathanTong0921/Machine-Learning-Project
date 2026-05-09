# Final Report Analysis Themes

## 1. 为什么 GBDT 取得最高分数

当前最高的单次 boosted-tree 结果来自 `gbdt_log_loss`：

| source | model | seed | accuracy | macro-F1 | AUC | dropout F1 | enrolled F1 | graduate F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `models/metrics/boost/classic_boosting_cross_seed_metrics.csv` | `gbdt_log_loss` | 62 | 0.7937 | 0.7501 | 0.9058 | 0.8144 | 0.5638 | 0.8723 |

需要在汇报里说明：这个结果来自 cross-seed 实验中的 seed 62，高于很多固定划分实验，但跨 split 比较时要标注评估口径。

GBDT 取得高分的核心原因：

- 该数据是典型表格数据，包含强非线性阈值关系，例如学期通过数、成绩、缴费状态、年龄、申请路径等。
- GBDT 逐轮拟合前一轮残差，能对少数困难样本区域做局部修正，比单棵树、线性模型更适合这种混合机制数据。
- `log_loss` 多分类目标直接优化概率质量分布，通常比只优化硬分类更利于 AUC 和后续 ensemble。
- Enrolled 类处在 Dropout 和 Graduate 中间，边界模糊；GBDT 的局部阈值组合可以比线性模型更好地捕捉这类非线性过渡区。

建议讲法：

> GBDT 的优势不是“树更强”，而是它把学生轨迹拆成了多个局部风险区域，并通过残差迭代不断修补最难的 Enrolled/Graduate 边界。

## 2. 什么 ensemble 取得最佳成果

最适合作为最终汇报主 ensemble 的结果是 OOF probability ensemble with semester members：

| source | model | accuracy | macro-F1 | AUC | dropout F1 | enrolled F1 | graduate F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `models/metrics/ensemble/oof_probability_ensemble_semester_members_metrics.csv` | `oof_probability_ensemble_semester_members_test` | 0.7824 | 0.7343 | 0.9013 | 0.7867 | 0.5487 | 0.8674 |

对比其他 ensemble：

| source | model | accuracy | macro-F1 | AUC | note |
| --- | --- | ---: | ---: | ---: | --- |
| `models/metrics/ensemble/oof_probability_ensemble_metrics.csv` | `oof_probability_ensemble_test` | 0.7869 | 0.7320 | 0.9041 | AUC 更高，macro-F1 略低 |
| `models/metrics/ensemble/machine_only_ensemble_metrics.csv` | `machine_only_ensemble_test` | 0.7696 | 0.7247 | 0.8982 | 纯 ML 版本，稳但不最高 |
| `models/metrics/ensemble/cross_seed_stacking_metrics.csv` | `cross_seed_soft_average_test` | n/a | 0.7247 | n/a | test 未超过 OOF semester ensemble |
| `models/metrics/ensemble/cross_seed_stacking_metrics.csv` | `cross_seed_stacking_test` | n/a | 0.7172 | n/a | stacking 在 held-out 上退化 |

最终推荐口径：

- 如果讲“稳定、可复现、无泄漏的最终 ensemble”，使用 `oof_probability_ensemble_semester_members_test`。
- 如果讲“单次最高分探索”，可以引用 `gbdt_log_loss` seed 62，但要说明它来自 cross-seed split。
- 如果讲“为什么不用 stacking”，强调 stacking OOF 看似不错，但 held-out test 下降，说明二层模型容易过拟合验证分布。

建议讲法：

> 最终 ensemble 不是简单堆模型，而是用 OOF 概率做权重搜索，再在 held-out test 上一次性评估。它的价值是降低泄漏风险，并把不同模型对 Enrolled 过渡区的判断进行概率融合。

## 3. MLP 的不同编码方式带来的影响

重点实验来自 `models/metrics/mlp/multiview_application_order_encoding_tuned_summary.csv`。

| encoding | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar tuned | 0.7718 | 0.7285 | 0.8889 | 0.7963 | 0.5882 | 0.8220 |
| thermometer tuned | 0.7824 | 0.7350 | 0.8903 | 0.7447 | 0.5798 | 0.8793 |
| target encoding OOF tuned | 0.7681 | 0.7299 | 0.8949 | 0.6956 | 0.6765 | 0.8477 |

关键结论：

- `Application order` 直接作为 scalar 会隐含等距假设，即第 1 志愿到第 2 志愿与第 8 到第 9 志愿的距离相同。
- Thermometer encoding 更适合表达有序但不严格等距的志愿顺序，重调参后 macro-F1 从 0.7285 提到 0.7350。
- Target encoding OOF 对 AUC 和 Enrolled acc 更友好，但牺牲 Dropout acc，说明它更像概率排序增强，而不是硬分类最优。
- 未调参直接替换编码曾经变差，说明“编码方式”和“超参数”存在交互，不能只换表示不重新搜索。

建议讲法：

> MLP 的提升来自改变特征假设：不是把志愿顺序当作连续数轴，而是让模型看到更合理的有序结构。

## 4. Course 171 outlier / subgroup

主要文档：`docs/findings/course_171_ctesp_creditation_subgroup.md`。

核心事实：

- `Course == 171` 对应 `Design de Animação e Multimédia`。
- `Course == 171` 且 12 个 curricular activity 字段全为 0 的样本共有 180 个：

| class | count |
| --- | ---: |
| Graduate | 75 |
| Dropout | 77 |
| Enrolled | 28 |

机制解释：

- 这个子群不像普通缺失值，更像 CTeSP-to-bachelor 或 credit recognition 路径造成的制度性特征形态。
- 学生可能因为已有课程/学分抵免，导致一年级活动列为 0，但并不代表没有学习轨迹。

为什么不能硬规则覆盖：

- 180 人中 Graduate 只有 75，不是纯 Graduate 群。
- CatBoost 对 75 个 true Graduate 子群也只预测 40 个 Graduate，平均 Graduate probability 约 0.371。
- `/tmp` 的 SVM 小样本实验中，holdout macro-F1 只有约 0.62，简单规则最好也只有约 0.69，稳定性不足。

推荐结论：

- 不做 `if Course171 zero curricular -> Graduate` 的硬覆盖。
- 更适合作为机制解释、uncertainty/report-only 标记、soft probability adjustment 或二阶段决策中的低风险辅助特征。

建议讲法：

> 这不是一个可以直接加分的硬规则，而是一个很有价值的机制发现：模型发现了一个制度路径子群，但这个子群内部仍然混合了三类学生。

## 5. SVM 去噪提升

主要文档：`docs/feature/svm_noise_feature_engineering.md`。

核心对比：

| model | accuracy | macro-F1 | AUC | enrolled recall | enrolled F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense SVM best | 0.7809 | 0.7249 | 0.8916 | 0.5168 | 0.5268 |
| dense drop-noise SVM best | 0.7854 | 0.7306 | 0.8910 | 0.5294 | 0.5396 |
| delta | +0.0045 | +0.0057 | -0.0006 | +0.0126 | +0.0128 |

去除的低价值/噪声特征：

- `Curricular units 1st sem (credited)`
- `Curricular units 2nd sem (credited)`
- `Curricular units 1st sem (without evaluations)`
- `Curricular units 2nd sem (without evaluations)`
- `International`
- `Educational special needs`

最优配置：

- feature set: `dense_drop_noise`
- kernel: `rbf`
- `C = 15.0`
- `gamma = 0.006`
- class weights: Dropout 1.0, Enrolled 1.25, Graduate 1.0

关键解释：

- SVM 尤其 RBF SVM 依赖距离度量，低信息维度会稀释真正有效的距离结构。
- 去噪后，Enrolled 的 recall 和 F1 提升更明显，说明边界模糊类更容易受噪声维度影响。
- 后续 SelectKBest 没有继续提升，说明本次收益来自有业务解释的目标去噪，而不是机械特征筛选。

建议讲法：

> 对树模型来说弱特征通常只是少用，对 SVM 来说弱特征会进入距离计算，因此“删掉坏维度”本身就是模型改进。
