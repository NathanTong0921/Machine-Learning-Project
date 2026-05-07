# optimize_two_stage_decision_thresholds.py

## 作用

对应脚本：`scripts/ensemble/optimize_two_stage_decision_thresholds.py`

这个脚本在已有 cumulative two-stage decomposition 上追加阈值校准，不覆盖原始 one-stage 或未校准 two-stage 结果。

## 背景动机

前面的 `search_cumulative_link_models.py` 已经验证了一个有序分类拆解：

```text
Dropout < Enrolled < Graduate
```

直觉上，这个任务不是完全无序三分类。根据论文定义，`Enrolled` 是正常学制结束时仍在读/未完成的中间状态，因此可以拆成两个递进判断：

1. 是否已经跌到最底线：`Dropout` vs `NotDropout`
2. 在没有辍学的前提下是否顺利毕业：`Enrolled` vs `Graduate`

未校准 two-stage 的问题是：把两个二分类概率重组为三分类概率后直接 argmax，会压低中间类 `Enrolled`。之前结果中：

| model | macro-F1 | Enrolled acc |
| --- | ---: | ---: |
| two_stage_uncalibrated_test | 0.7083 | 0.4748 |

这说明二分类器本身可分，但最终三分类决策规则对中间类不友好。进一步的 Enrolled feature-space side-analysis 也支持这一点：约 27.8% 的真实 `Enrolled` 会在 Stage A 的线性边界上落到 `Dropout` 一侧，容易被第一阶段提前截断。

因此本实验的核心不是换模型，而是校准二阶段决策阈值，让更多边界样本进入 Stage B，并给 `Enrolled` 留出更宽的判定空间。

二阶段仍然是：

- Stage A：`Dropout` vs `NotDropout`
- Stage B：`Enrolled` vs `Graduate`

不同点是最终预测不再简单取三分类 argmax，而是在 OOF 上搜索两个阈值：

- `stage_a_threshold_not_dropout`
- `stage_b_threshold_graduate`

预测规则：

```text
if P(NotDropout) < stage_a_threshold_not_dropout:
    predict Dropout
else:
    if P(Graduate | NotDropout) >= stage_b_threshold_graduate:
        predict Graduate
    else:
        predict Enrolled
```

阈值只用 OOF macro-F1 选择，测试集只最后评估一次。

## 实验设计

### 成员模型

本脚本复用 cumulative two-stage 的三个传统机器学习成员：

| member | Stage A | Stage B | 参数来源 |
| --- | --- | --- | --- |
| `catboost_cumulative` | CatBoost binary | CatBoost binary | `catboost_tuned_best_params.json` |
| `lightgbm_hist_cumulative` | LightGBM binary | LightGBM binary | `lightgbm_tuned_best_params.json` + hist column best |
| `gbdt_cumulative` | sklearn GBDT binary | sklearn GBDT binary | `classic_boosting_best_params.json` |

Stage A 使用全部训练样本：

```text
Dropout -> Dropout
Enrolled, Graduate -> NotDropout
```

Stage B 只用非 Dropout 训练样本：

```text
Enrolled -> Enrolled
Graduate -> Graduate
```

### 概率重组

每个成员先输出两个二分类概率：

```text
P(NotDropout)
P(Graduate | NotDropout)
```

未校准版本会重组为三分类概率：

```text
P(Dropout)  = 1 - P(NotDropout)
P(Enrolled) = P(NotDropout) * (1 - P(Graduate | NotDropout))
P(Graduate) = P(NotDropout) * P(Graduate | NotDropout)
```

然后直接 argmax。

阈值校准版本不直接 argmax，而是使用两个阈值：

```text
if P(NotDropout) < t_A:
    predict Dropout
elif P(Graduate | NotDropout) >= t_B:
    predict Graduate
else:
    predict Enrolled
```

这里：

- `t_A` 越低，越不容易判 Dropout，更多样本进入 Stage B。
- `t_B` 越高，越不容易判 Graduate，`Enrolled` 的判定空间更大。

### OOF 选择流程

为了避免在测试集上调参，脚本使用 5-fold OOF：

1. 每个成员在 5-fold 上生成 OOF 三分类概率。
2. 用 OOF macro-F1 搜索成员 ensemble 权重，步长为 `0.05`。
3. 把 OOF ensemble 概率还原成：
   - `P(NotDropout)`
   - `P(Graduate | NotDropout)`
4. 在 OOF 上搜索 `t_A` 和 `t_B`：
   - 先粗搜 `0.20` 到 `0.80`
   - 再围绕最优阈值做小范围细搜，步长 `0.005`
5. 固定权重和阈值后，只在测试集评估一次。

选择指标是 OOF `macro-F1`，因为 `Enrolled` 是最小且最难的类，accuracy 容易偏向 `Graduate`。

## 输出

- `models/metrics/two_stage_threshold_tuned_metrics.csv`
- `models/metrics/two_stage_vs_one_stage_comparison.csv`
- `models/metrics/two_stage_threshold_tuned_weight_grid.csv`
- `models/metrics/two_stage_threshold_tuned_threshold_grid.csv`
- `models/metrics/two_stage_threshold_tuned_best.json`
- `reports/figures/two_stage_vs_one_stage_comparison.png`

## 当前结果

正式运行命令：

```bash
python scripts/ensemble/optimize_two_stage_decision_thresholds.py --cv 5 --n-jobs 96
```

OOF 选择出的成员权重：

| member | weight |
| --- | ---: |
| catboost_cumulative | 0.30 |
| lightgbm_hist_cumulative | 0.40 |
| gbdt_cumulative | 0.30 |

OOF 选择出的阈值：

| threshold | value |
| --- | ---: |
| `stage_a_threshold_not_dropout` | 0.355 |
| `stage_b_threshold_graduate` | 0.550 |

阈值解释：

- `t_A = 0.355`：只要 `P(NotDropout)` 达到 0.355 就不急着判 Dropout。这比默认 0.5 更保守地使用 Dropout 标签，符合“避免 Stage A 误杀 Enrolled”的目标。
- `t_B = 0.550`：只有 `P(Graduate | NotDropout)` 达到 0.55 才判 Graduate。相比默认 0.5，这给 `Enrolled` 多留了一些空间。

测试集结果：

| model | accuracy | macro-F1 | weighted-F1 | FMI | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| two_stage_uncalibrated_test | 0.7658 | 0.7083 | 0.7648 | 0.6803 | 0.8925 | 0.8056 | 0.4748 | 0.8446 |
| two_stage_threshold_tuned_test | 0.7628 | 0.7217 | 0.7709 | 0.6717 | 0.8925 | 0.7330 | 0.6176 | 0.8341 |

校准带来的变化：

| metric | uncalibrated | threshold tuned | delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.7658 | 0.7628 | -0.0030 |
| macro-F1 | 0.7083 | 0.7217 | +0.0134 |
| weighted-F1 | 0.7648 | 0.7709 | +0.0061 |
| Dropout acc | 0.8056 | 0.7330 | -0.0726 |
| Enrolled acc | 0.4748 | 0.6176 | +0.1429 |
| Graduate acc | 0.8446 | 0.8341 | -0.0106 |

可以看到，校准的主要收益来自 `Enrolled`。它牺牲了一部分 Dropout recall，把原来过早判 Dropout 的边界样本放回二阶段判断，从而大幅提高了 Enrolled recall。

与 one-stage 参考对比：

| model | type | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final OOF ensemble with advanced/tree-semester members | one-stage/mixed ensemble | 0.7824 | 0.7343 | 0.9013 | 0.7471 | 0.5798 | 0.8778 |
| machine_only_ensemble_test | one-stage machine ensemble | 0.7696 | 0.7247 | 0.8982 | 0.7330 | 0.5924 | 0.8567 |
| catboost_tuned | one-stage CatBoost | 0.7590 | 0.7222 | 0.8959 | 0.7260 | 0.6134 | 0.8235 |
| two_stage_threshold_tuned_test | calibrated two-stage | 0.7628 | 0.7217 | 0.8925 | 0.7330 | 0.6176 | 0.8341 |
| lightgbm_hist_column_best | one-stage LightGBM hist | 0.7718 | 0.7196 | 0.8921 | 0.7471 | 0.5336 | 0.8733 |

相对差值表如下。这里每个数值都按：

```text
two_stage_threshold_tuned_test - one_stage_reference
```

计算，所以正数表示 calibrated two-stage 更高，负数表示 calibrated two-stage 更低。

| model | type | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vs final OOF ensemble with advanced/tree-semester members | calibrated two-stage minus one-stage/mixed ensemble | -0.0196 | -0.0126 | -0.0087 | -0.0141 | +0.0378 | -0.0437 |
| vs machine_only_ensemble_test | calibrated two-stage minus one-stage machine ensemble | -0.0068 | -0.0030 | -0.0057 | +0.0000 | +0.0252 | -0.0226 |
| vs catboost_tuned | calibrated two-stage minus one-stage CatBoost | +0.0038 | -0.0005 | -0.0034 | +0.0070 | +0.0042 | +0.0106 |
| vs lightgbm_hist_column_best | calibrated two-stage minus one-stage LightGBM hist | -0.0090 | +0.0021 | +0.0004 | -0.0141 | +0.0840 | -0.0392 |

结论：

- 阈值校准明显优于未校准 two-stage：test macro-F1 从 0.7083 提升到 0.7217，`Enrolled` accuracy 从 0.4748 提升到 0.6176。
- 但它没有超过当前最强 one-stage / machine-only ensemble：machine-only ensemble test macro-F1 是 0.7247，AUC 是 0.8982。
- 它与 tuned CatBoost 很接近，但仍略低：0.7217 vs 0.7222。
- 它的价值在于提高 `Enrolled` 识别，而不是刷新总体最好分数。代价是 Dropout/Graduate 的 recall 被压低，整体 accuracy 也低于 machine-only ensemble。

## 是否优于 One-Stage

答案分两层：

### 不是总体最优

如果按主指标 test macro-F1 排序：

```text
final mixed OOF ensemble      0.7343
machine-only one-stage        0.7247
tuned CatBoost one-stage      0.7222
tuned two-stage               0.7217
LightGBM hist one-stage       0.7196
uncalibrated two-stage        0.7083
```

所以 tuned two-stage 没有超过当前最好的 one-stage / ensemble。

### 但更偏向 Enrolled

如果重点是识别 `Enrolled`：

```text
tuned two-stage Enrolled acc        0.6176
machine-only ensemble Enrolled acc  0.5924
LightGBM hist Enrolled acc          0.5336
uncalibrated two-stage Enrolled acc 0.4748
```

因此 tuned two-stage 是一个“更照顾中间类”的模型，而不是一个整体最强模型。

## 解释和启发

这个结果和可视化/距离分析一致：

- `Enrolled` 不是干净独立大簇，而是位于 `Dropout` 和 `Graduate` 之间。
- 直接 two-stage argmax 会让中间类被两端吃掉。
- 降低 Stage A 阈值可以减少把 `Enrolled` 过早判成 `Dropout`。
- 提高 Stage B 阈值可以减少把边界 `Enrolled` 过早判成 `Graduate`。

但是，阈值校准仍然没有刷新总分，说明 `Enrolled` 的边界不是简单通过两个固定阈值就能完全解决。它更像一个具有内部子群的过渡标签：

- 一部分接近 Dropout
- 一部分是 enrolled-core
- 一部分接近 Graduate
- 一部分位于 Enrolled/Graduate 边界

## 报告中建议如何写

建议把这个实验作为 side-analysis / model-design ablation，而不是主模型：

```text
We also tested an ordinal two-stage decomposition matching the natural ordering
Dropout < Enrolled < Graduate. Direct recomposition underperformed because the
middle class was suppressed. OOF threshold tuning recovered much of the Enrolled
recall and improved macro-F1 from 0.7083 to 0.7217, but it still did not surpass
the strongest one-stage ensemble. This suggests that Enrolled is best treated as
a true middle state rather than collapsed into either endpoint.
```

中文报告可以写：

```text
我们进一步尝试了符合标签语义的二阶段有序分类策略。未校准版本会明显压低
Enrolled 类；通过 OOF 阈值搜索后，Enrolled accuracy 从 0.4748 提升到
0.6176，macro-F1 从 0.7083 提升到 0.7217。但该结果仍略低于 one-stage
machine-only ensemble 的 0.7247，说明二阶段结构有解释价值，但不是最终最佳预测策略。
```

## 后续方向

如果继续挖 two-stage，可以优先考虑：

- 对 Stage A 使用 recall-constrained tuning，显式约束 `Enrolled` 被判成 `Dropout` 的比例。
- 对 Stage B 使用 class-specific calibration，而不是单一全局阈值。
- 把 Enrolled distance tags 加入 meta-model，而不是只用两个二分类概率。
- 用 CatBoost/RF leaf embedding 或 supervised UMAP embedding 作为 stacking side features。

当前不建议：

- 直接把 `Enrolled` 合并到 `Graduate`。
- 直接把 `Enrolled` 合并到 `Dropout`。
- 用未校准 two-stage 作为主结果。

## 运行方式

```bash
python scripts/ensemble/optimize_two_stage_decision_thresholds.py --cv 5 --n-jobs 96
```
