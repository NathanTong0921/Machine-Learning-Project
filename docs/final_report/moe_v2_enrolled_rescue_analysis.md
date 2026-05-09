# MoE v2 Enrolled Rescue Analysis

## 中文结论

这次分析的目标不是继续提分，而是定位 MoE v2 为什么能提高 `Enrolled recall`。

核心结论：

- MoE v2 在 test set 中正确识别了 `145 / 238` 个真实 `Enrolled`，recall = `0.6092`。
- 相比 `lightgbm_hist_column_best`，MoE v2 额外救回了 `24` 个真实 `Enrolled`，占全部真实 Enrolled 的 `10.08%`。
- 但 `rescued_vs_all_experts = 0`，说明 MoE v2 **没有凭空救回三个专家全都不投 Enrolled 的样本**。
- 它的实际作用是：在专家分歧时，降低 LightGBM 的主导权，同时给 CatBoost / GBDT 的 Enrolled 信号保留足够权重，使 soft probability 的 argmax 转回 `Enrolled`。

因此，MoE v2 改善 Enrolled recall 的机制不是“router 独立发现了新类别边界”，而是：

> Router 在边界样本上识别专家适配性，把 LightGBM 容易压成 Dropout/Graduate 的样本重新融合回 Enrolled。

最适合汇报的解释对象是 `rescued_vs_lightgbm_hist_column_best` 这 24 个样本，而不是 `rescued_vs_all_experts`。

这些被 LightGBM 漏掉、MoE 救回的样本画像：

- 平均年龄更低：`20.67`，低于全部 Enrolled 的 `22.64`。
- 入学成绩更低：Admission grade 平均 `122.16`，低于全部 Enrolled 的 `126.06`。
- 第一、第二学期 approved 数更低：`3.42 / 2.92`，低于全部 Enrolled 的 `4.13 / 3.89`。
- scholarship holder 比例更高：`0.2917`，高于全部 Enrolled 的 `0.1639`。
- MoE 给这些样本的平均 Enrolled probability 是 `0.5116`，刚好是边界附近，不是高置信度样本。

Router 行为也支持这个解释：

- 全部 test 样本中，LightGBM 平均 router weight 是 `0.3599`。
- 真实 Enrolled 中，LightGBM 平均 weight 降到 `0.2795`。
- MoE 正确识别的 Enrolled 中，LightGBM 平均 weight 进一步降到 `0.2575`。
- `rescued_vs_lightgbm` 中，LightGBM 平均 weight 只有 `0.2498`，CatBoost 平均 weight 达到 `0.4469`。

这说明 router 确实学到：对于一部分 Enrolled 边界样本，LightGBM 不是最可靠专家，CatBoost/GBDT 的概率更值得保留。

## Router 工作原理

MoE v2 的 router 不是普通三分类器。它不直接预测学生是 `Dropout / Enrolled / Graduate`，而是预测：

> 当前样本应该更相信哪个 expert。

### 1. Expert Pool

当前 MoE v2 使用三个传统 ML expert：

| expert id | expert name | 角色 |
| ---: | --- | --- |
| 0 | `catboost_tuned` | 类别变量和类别-数值交互较强 |
| 1 | `lightgbm_hist_column_best` | 稳定的强 boosted-tree expert |
| 2 | `gbdt_log_loss` | 对部分局部边界和 Enrolled 样本有补充 |

每个 expert 都输出三类概率：

```text
expert_k(x) = [P(Dropout), P(Enrolled), P(Graduate)]
```

### 2. Router 训练标签怎么来

训练 router 时，不能让 expert 在自己训练过的样本上给概率，否则会高估 expert 能力。因此先做 5-fold OOF：

1. 把 train split 成 5 折。
2. 每个 expert 在 4 折上训练，在剩下一折上预测概率。
3. 拼起来得到每个训练样本的 OOF expert probabilities。

对每个训练样本 `i`，已知真实标签 `y_i`。计算每个 expert 对真实标签的 negative log loss：

```text
loss_k(i) = -log P_k(y_i | x_i)
```

哪个 expert 对真实标签给出的概率最高，loss 就最低。于是 router 的 pseudo-label 是：

```text
router_label(i) = argmin_k loss_k(i)
```

所以 router 学的是：

```text
x_i -> 在这个样本上哪个 expert 最靠谱
```

这就是 supervised error-predicting router 的含义：监督信号来自真实标签下的 expert error，而不是原始学生类别。

### 3. Router 输入是什么

当前 best 配置是：

```text
router_feature_variant = raw
router = LogisticRegression(class_weight="balanced")
weight_mode = none
```

也就是 router 直接读取原始 36 列表格特征，经 dense preprocessing 后训练：

- numeric columns: median impute + standard scaling
- categorical columns: most-frequent impute + one-hot

然后 multinomial logistic regression 输出三个 expert 权重：

```text
router(x) = [g_catboost(x), g_lightgbm(x), g_gbdt(x)]
```

这三个值是 softmax 概率，表示各 expert 对当前样本的适配程度。

### 4. 推理阶段如何合成最终概率

对一个 test 样本 `x`：

1. 三个 experts 分别输出类别概率：

```text
CatBoost(x) = p_0
LightGBM(x) = p_1
GBDT(x) = p_2
```

2. router 输出 expert 权重：

```text
router(x) = [g_0, g_1, g_2]
```

3. soft routing 做加权平均：

```text
P_final(y | x) = g_0 * p_0(y) + g_1 * p_1(y) + g_2 * p_2(y)
```

最后取最大概率类别：

```text
prediction = argmax_y P_final(y | x)
```

因此 router 不会直接说“这个学生是 Enrolled”。它只是决定每个 expert 的话语权。最终是否预测 `Enrolled`，取决于融合后的 `P_final(Enrolled)` 是否超过 `Dropout / Graduate`。

### 5. 为什么这能改善 Enrolled Recall

`Enrolled` 是中间态类别，很多样本位于 `Dropout` 和 `Graduate` 的边界。单个强模型通常会形成全局偏置：

- LightGBM 对一些边界样本更容易压向 `Dropout` 或 `Graduate`。
- CatBoost / GBDT 有时会给这些样本更高的 `Enrolled` probability。

MoE v2 的 router 观察到某些特征区域中，LightGBM 不是最低 loss expert，于是降低 LightGBM 权重，提高 CatBoost / GBDT 权重。

这次 rescue analysis 正好验证了这一点：

| group | CatBoost weight | LightGBM weight | GBDT weight |
| --- | ---: | ---: | ---: |
| all test | 0.3082 | 0.3599 | 0.3319 |
| true Enrolled | 0.4015 | 0.2795 | 0.3191 |
| MoE correct Enrolled | 0.4345 | 0.2575 | 0.3080 |
| rescued vs LightGBM | 0.4469 | 0.2498 | 0.3033 |

也就是说，在真实 `Enrolled` 样本上，router 系统性地降低 LightGBM 权重；在 `rescued_vs_lightgbm` 的 24 个样本里，这个趋势更明显。

### 6. 具体样本例子

以样本 `test_row=777` 为例：

| model | prediction | P(Enrolled) | router weight |
| --- | --- | ---: | ---: |
| CatBoost | Enrolled | 0.6735 | 0.4743 |
| LightGBM | Dropout | 0.3363 | 0.0453 |
| GBDT | Enrolled | 0.8426 | 0.4804 |
| MoE v2 | Enrolled | 0.7394 | - |

LightGBM 单独会错成 `Dropout`，但 router 几乎不给它权重，只保留 CatBoost 和 GBDT 的判断，所以最终 MoE v2 预测回 `Enrolled`。

这就是 MoE v2 的可解释价值：它不是黑盒平均，而是给出了“为什么这个样本要听哪个专家”的局部解释。

This analysis reproduces the current MoE v2 best configuration:

- experts: `catboost_tuned`, `lightgbm_hist_column_best`, `gbdt_log_loss`
- router: `logistic_router`
- router features: raw tabular features
- aggregation: soft routing

## Main Counts

MoE v2 predicts `Enrolled` correctly for **145 / 238** true-Enrolled test samples.

Most relevant rescue counts:

- rescued vs `lightgbm_hist_column_best`: **24**
- rescued vs all three experts: **0**

| group                                      |   count |   share_of_true_enrolled |
|:-------------------------------------------|--------:|-------------------------:|
| true_enrolled_total                        |     238 |                   1.0000 |
| moe_correct_enrolled                       |     145 |                   0.6092 |
| moe_missed_enrolled                        |      93 |                   0.3908 |
| catboost_tuned_correct_enrolled            |     146 |                   0.6134 |
| rescued_vs_catboost_tuned                  |      14 |                   0.0588 |
| lightgbm_hist_column_best_correct_enrolled |     127 |                   0.5336 |
| rescued_vs_lightgbm_hist_column_best       |      24 |                   0.1008 |
| gbdt_log_loss_correct_enrolled             |     147 |                   0.6176 |
| rescued_vs_gbdt_log_loss                   |      13 |                   0.0546 |
| rescued_vs_all_experts                     |       0 |                   0.0000 |
| all_experts_missed_enrolled                |      65 |                   0.2731 |

## Router Behavior

| group                  |    n |   mean_weight_catboost_tuned |   top_route_share_catboost_tuned |   mean_weight_lightgbm_hist_column_best |   top_route_share_lightgbm_hist_column_best |   mean_weight_gbdt_log_loss |   top_route_share_gbdt_log_loss |
|:-----------------------|-----:|-----------------------------:|---------------------------------:|----------------------------------------:|--------------------------------------------:|----------------------------:|--------------------------------:|
| all_test               | 1328 |                       0.3082 |                           0.2937 |                                  0.3599 |                                      0.4639 |                      0.3319 |                          0.2425 |
| all_true_enrolled      |  238 |                       0.4015 |                           0.5210 |                                  0.2795 |                                      0.2689 |                      0.3191 |                          0.2101 |
| moe_correct_enrolled   |  145 |                       0.4345 |                           0.6069 |                                  0.2575 |                                      0.2000 |                      0.3080 |                          0.1931 |
| moe_missed_enrolled    |   93 |                       0.3500 |                           0.3871 |                                  0.3136 |                                      0.3763 |                      0.3364 |                          0.2366 |
| rescued_vs_lightgbm    |   24 |                       0.4469 |                           0.5417 |                                  0.2498 |                                      0.2083 |                      0.3033 |                          0.2500 |
| rescued_vs_all_experts |    0 |                     nan      |                         nan      |                                nan      |                                    nan      |                    nan      |                        nan      |

## Feature Profile

The table below compares true-Enrolled groups. Use `rescued_vs_lightgbm` as the main interpretation group, because LightGBM hist-column is the strongest stable single expert in the MoE pool.

| group                  |    n |   Application order mean |   Application order median |   Application mode mean |   Application mode median |   Course mean |   Course median |   Previous qualification (grade) mean |   Previous qualification (grade) median |   Admission grade mean |   Admission grade median |   Age at enrollment mean |   Age at enrollment median |   Tuition fees up to date mean |   Tuition fees up to date median |   Debtor mean |   Debtor median |   Scholarship holder mean |   Scholarship holder median |   Curricular units 1st sem (enrolled) mean |   Curricular units 1st sem (enrolled) median |   Curricular units 1st sem (approved) mean |   Curricular units 1st sem (approved) median |   Curricular units 1st sem (grade) mean |   Curricular units 1st sem (grade) median |   Curricular units 2nd sem (enrolled) mean |   Curricular units 2nd sem (enrolled) median |   Curricular units 2nd sem (approved) mean |   Curricular units 2nd sem (approved) median |   Curricular units 2nd sem (grade) mean |   Curricular units 2nd sem (grade) median |   router_weight_catboost_tuned mean |   router_weight_lightgbm_hist_column_best mean |   router_weight_gbdt_log_loss mean |   moe_v2_prob_enrolled mean |
|:-----------------------|-----:|-------------------------:|---------------------------:|------------------------:|--------------------------:|--------------:|----------------:|--------------------------------------:|----------------------------------------:|-----------------------:|-------------------------:|-------------------------:|---------------------------:|-------------------------------:|---------------------------------:|--------------:|----------------:|--------------------------:|----------------------------:|-------------------------------------------:|---------------------------------------------:|-------------------------------------------:|---------------------------------------------:|----------------------------------------:|------------------------------------------:|-------------------------------------------:|---------------------------------------------:|-------------------------------------------:|---------------------------------------------:|----------------------------------------:|------------------------------------------:|------------------------------------:|-----------------------------------------------:|-----------------------------------:|----------------------------:|
| all_true_enrolled      |  238 |                   1.6975 |                     1.0000 |                 20.6345 |                   17.0000 |     8829.3277 |       9147.0000 |                              131.1601 |                                130.0000 |               126.0580 |                 124.3500 |                  22.6387 |                    20.0000 |                         0.9454 |                           1.0000 |        0.1387 |          0.0000 |                    0.1639 |                      0.0000 |                                     5.9454 |                                       6.0000 |                                     4.1261 |                                       4.0000 |                                 10.9521 |                                   12.0000 |                                     5.8992 |                                       6.0000 |                                     3.8866 |                                       4.0000 |                                 10.9328 |                                   12.0000 |                              0.4015 |                                         0.2795 |                             0.3191 |                      0.4923 |
| moe_correct_enrolled   |  145 |                   1.6000 |                     1.0000 |                 21.8276 |                   17.0000 |     8989.8138 |       9238.0000 |                              131.0359 |                                131.0000 |               125.5441 |                 124.1000 |                  22.4483 |                    20.0000 |                         0.9586 |                           1.0000 |        0.1448 |          0.0000 |                    0.1310 |                      0.0000 |                                     5.8207 |                                       6.0000 |                                     3.9517 |                                       4.0000 |                                 11.2357 |                                   11.8333 |                                     5.7724 |                                       6.0000 |                                     3.7862 |                                       4.0000 |                                 11.5754 |                                   12.0000 |                              0.4345 |                                         0.2575 |                             0.3080 |                      0.6440 |
| moe_missed_enrolled    |   93 |                   1.8495 |                     1.0000 |                 18.7742 |                   17.0000 |     8579.1075 |       9147.0000 |                              131.3538 |                                130.0000 |               126.8591 |                 124.7000 |                  22.9355 |                    20.0000 |                         0.9247 |                           1.0000 |        0.1290 |          0.0000 |                    0.2151 |                      0.0000 |                                     6.1398 |                                       6.0000 |                                     4.3978 |                                       5.0000 |                                 10.5100 |                                   12.1250 |                                     6.0968 |                                       6.0000 |                                     4.0430 |                                       5.0000 |                                  9.9309 |                                   11.6857 |                              0.3500 |                                         0.3136 |                             0.3364 |                      0.2557 |
| rescued_vs_lightgbm    |   24 |                   1.8750 |                     1.0000 |                 18.6250 |                   17.0000 |     8625.2917 |       9254.0000 |                              129.0500 |                                128.5000 |               122.1625 |                 122.7000 |                  20.6667 |                    20.0000 |                         0.9167 |                           1.0000 |        0.1667 |          0.0000 |                    0.2917 |                      0.0000 |                                     5.4167 |                                       6.0000 |                                     3.4167 |                                       4.0000 |                                  9.5968 |                                   11.7500 |                                     5.4167 |                                       6.0000 |                                     2.9167 |                                       3.0000 |                                 10.0701 |                                   11.7333 |                              0.4469 |                                         0.2498 |                             0.3033 |                      0.5116 |
| rescued_vs_all_experts |    0 |                 nan      |                   nan      |                nan      |                  nan      |      nan      |        nan      |                              nan      |                                nan      |               nan      |                 nan      |                 nan      |                   nan      |                       nan      |                         nan      |      nan      |        nan      |                  nan      |                    nan      |                                   nan      |                                     nan      |                                   nan      |                                     nan      |                                nan      |                                  nan      |                                   nan      |                                     nan      |                                   nan      |                                     nan      |                                nan      |                                  nan      |                            nan      |                                       nan      |                           nan      |                    nan      |
| true_not_enrolled      | 1090 |                   1.7633 |                     1.0000 |                 19.0826 |                   17.0000 |     8931.6615 |       9254.0000 |                              132.5071 |                                133.1000 |               127.3563 |                 126.8000 |                  23.5404 |                    20.0000 |                         0.8661 |                           1.0000 |        0.1183 |          0.0000 |                    0.2550 |                      0.0000 |                                     6.4119 |                                       6.0000 |                                     4.8587 |                                       5.0000 |                                 10.6274 |                                   12.3333 |                                     6.3927 |                                       6.0000 |                                     4.6266 |                                       5.0000 |                                 10.2200 |                                   12.2727 |                              0.2878 |                                         0.3774 |                             0.3348 |                      0.1878 |

## Interpretation

The router improves Enrolled recall mainly by shifting probability mass on borderline true-Enrolled samples. These samples are often not cleanly assigned by the strongest tree expert, but the soft router keeps nonzero weight on alternative experts and raises the final Enrolled probability enough to cross the decision boundary.

This should be presented as an interpretability result, not as a new performance claim: MoE v2 explains a subset of Enrolled recovery behavior, while the final leaderboard model remains the OOF probability ensemble / best boosted-tree results depending on the reporting口径.
