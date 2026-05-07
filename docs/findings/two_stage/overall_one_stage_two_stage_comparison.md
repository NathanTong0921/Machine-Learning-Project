# Overall One-Stage vs Two-Stage Model Comparison

## 目的

这份文档把当前项目里最重要的 one-stage、two-stage、machine-only ensemble 和 mixed ensemble 结果放在同一张视角下比较。

这里的 one-stage 指直接预测三分类：

```text
Dropout / Enrolled / Graduate
```

two-stage 指先做递进二分类，再组合成三分类：

```text
Stage A: Dropout vs NotDropout
Stage B: Enrolled vs Graduate
```

主比较指标是 `macro-F1`，因为 `Enrolled` 是少数类且最难预测。Accuracy 和 AUC 作为辅助指标；`Dropout acc / Enrolled acc / Graduate acc` 等价于对应类别 recall。

注意：部分早期 metrics CSV 没有保存 FMI 和 per-class accuracy，因此表中对应位置记为 `n/a`。

## 总览表

| group | model | accuracy | macro-F1 | weighted-F1 | FMI | AUC | Dropout acc | Enrolled acc | Graduate acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed ensemble | final OOF semester-members ensemble | 0.7824 | 0.7343 | 0.7843 | n/a | 0.9013 | 0.7471 | 0.5798 | 0.8778 |
| one-stage ensemble | machine-only OOF ensemble | 0.7696 | 0.7247 | 0.7732 | 0.6784 | 0.8982 | 0.7330 | 0.5924 | 0.8567 |
| one-stage ensemble | cross-seed soft average | 0.7673 | 0.7247 | 0.7727 | 0.6747 | 0.8976 | 0.7330 | 0.6092 | 0.8462 |
| one-stage | tuned CatBoost | 0.7590 | 0.7222 | 0.7683 | n/a | 0.8959 | 0.7260 | 0.6134 | 0.8235 |
| two-stage | threshold-tuned two-stage | 0.7628 | 0.7217 | 0.7709 | 0.6717 | 0.8925 | 0.7330 | 0.6176 | 0.8341 |
| advanced one-stage | MultiView Gated MLP | 0.7575 | 0.7217 | 0.7683 | n/a | 0.8925 | n/a | n/a | n/a |
| one-stage | LightGBM hist-column best | 0.7718 | 0.7196 | 0.7720 | n/a | 0.8921 | 0.7471 | 0.5336 | 0.8733 |
| one-stage | Random Forest | 0.7523 | 0.7136 | 0.7614 | n/a | 0.8852 | n/a | n/a | n/a |
| advanced one-stage | TabPFN | 0.7861 | 0.7131 | 0.7752 | n/a | 0.9017 | n/a | n/a | n/a |
| two-stage | uncalibrated two-stage | 0.7658 | 0.7083 | 0.7648 | 0.6803 | 0.8925 | 0.8056 | 0.4748 | 0.8446 |

## 分组结论

### 1. 总体最优仍是 mixed ensemble

当前最好结果仍是：

```text
final OOF semester-members ensemble
accuracy  = 0.7824
macro-F1  = 0.7343
AUC       = 0.9013
```

它用了更丰富的成员，包括 tuned tree models、TabPFN / MLP 类高级成员和 semester-tree 成员，因此不适合作为纯 machine-only 对照，但适合作为最终 best score。

### 2. Machine-only one-stage ensemble 是传统 ML 主结果

如果只看传统机器学习模型，当前推荐主结果是：

```text
machine-only OOF ensemble
accuracy     = 0.7696
macro-F1     = 0.7247
AUC          = 0.8982
Enrolled acc = 0.5924
```

它比单个 tuned CatBoost、LightGBM hist 和 Random Forest 更稳，且 AUC 明显更高。

### 3. Cross-seed soft average 和 machine-only ensemble 基本打平

Cross-seed soft average 的 macro-F1 也是约 0.7247，但 AUC 略低：

```text
cross-seed soft average AUC = 0.8976
machine-only ensemble AUC   = 0.8982
```

它的 `Enrolled acc` 更高一点：

```text
cross-seed soft average Enrolled acc = 0.6092
machine-only ensemble Enrolled acc   = 0.5924
```

但 machine-only OOF ensemble 的整体 AUC 和 Graduate recall 更好，因此更适合作为主表中的 machine-only ensemble。

### 4. Two-stage threshold tuning 有用，但不是总体最优

未校准 two-stage 的问题很明显：

```text
uncalibrated two-stage macro-F1     = 0.7083
uncalibrated two-stage Enrolled acc = 0.4748
```

阈值校准后：

```text
threshold-tuned two-stage macro-F1     = 0.7217
threshold-tuned two-stage Enrolled acc = 0.6176
```

提升幅度：

| metric | uncalibrated two-stage | threshold-tuned two-stage | delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.7658 | 0.7628 | -0.0030 |
| macro-F1 | 0.7083 | 0.7217 | +0.0134 |
| weighted-F1 | 0.7648 | 0.7709 | +0.0061 |
| Dropout acc | 0.8056 | 0.7330 | -0.0726 |
| Enrolled acc | 0.4748 | 0.6176 | +0.1429 |
| Graduate acc | 0.8446 | 0.8341 | -0.0106 |

这说明 threshold tuning 正在做它应该做的事：牺牲一部分 Dropout recall，换来更好的 `Enrolled` 识别。

### 5. Two-stage tuned vs one-stage machine-only

最关键的直接比较：

| model | accuracy | macro-F1 | AUC | Dropout acc | Enrolled acc | Graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| machine-only OOF ensemble | 0.7696 | 0.7247 | 0.8982 | 0.7330 | 0.5924 | 0.8567 |
| threshold-tuned two-stage | 0.7628 | 0.7217 | 0.8925 | 0.7330 | 0.6176 | 0.8341 |

差值：

```text
two-stage tuned - machine-only ensemble
accuracy      -0.0068
macro-F1      -0.0030
AUC           -0.0057
Dropout acc    0.0000
Enrolled acc  +0.0252
Graduate acc  -0.0226
```

解释：

- 如果目标是总体最佳预测，machine-only one-stage ensemble 更好。
- 如果目标是优先提高 `Enrolled` 召回，threshold-tuned two-stage 更有价值。
- Two-stage 的主要代价是 Graduate recall 和 AUC 下降。

## 相对差距表

### 以 Machine-Only One-Stage Ensemble 为基准

这个基准最适合回答“two-stage 是否比干净的 one-stage machine method 更好”，因为两者都不依赖深度学习架构。

基准：

```text
machine-only OOF ensemble
accuracy = 0.7696
macro-F1 = 0.7247
AUC      = 0.8982
```

相对差距定义：

```text
delta = compared_model - machine_only_ensemble
relative_delta = delta / machine_only_ensemble
```

| compared model | accuracy delta | accuracy rel. | macro-F1 delta | macro-F1 rel. | AUC delta | AUC rel. | Dropout acc delta | Enrolled acc delta | Graduate acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold-tuned two-stage | -0.0068 | -0.88% | -0.0030 | -0.42% | -0.0057 | -0.63% | 0.0000 | +0.0252 | -0.0226 |
| uncalibrated two-stage | -0.0038 | -0.49% | -0.0164 | -2.26% | -0.0057 | -0.63% | +0.0726 | -0.1176 | -0.0121 |
| cross-seed soft average | -0.0023 | -0.29% | -0.0001 | -0.01% | -0.0006 | -0.07% | 0.0000 | +0.0168 | -0.0106 |
| final mixed ensemble | +0.0128 | +1.66% | +0.0095 | +1.31% | +0.0031 | +0.34% | +0.0141 | -0.0126 | +0.0211 |

解读：

- `threshold-tuned two-stage` 相比 machine-only one-stage ensemble，macro-F1 低 `0.0030`，相对低 `0.42%`。
- 它唯一明显更强的是 `Enrolled acc`，高 `0.0252`。
- 它的代价是 `Graduate acc` 低 `0.0226`，AUC 低 `0.0057`。
- `uncalibrated two-stage` 虽然 Dropout acc 高很多，但 Enrolled acc 大幅下降，说明 naive two-stage 主要是把中间类压到两端。

### 以 Final Mixed Ensemble 为基准

这个基准回答“离当前最终最好结果还差多少”。

| compared model | accuracy delta | accuracy rel. | macro-F1 delta | macro-F1 rel. | AUC delta | AUC rel. | Dropout acc delta | Enrolled acc delta | Graduate acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| machine-only OOF ensemble | -0.0128 | -1.64% | -0.0095 | -1.30% | -0.0031 | -0.34% | -0.0141 | +0.0126 | -0.0211 |
| threshold-tuned two-stage | -0.0196 | -2.50% | -0.0126 | -1.71% | -0.0087 | -0.97% | -0.0141 | +0.0378 | -0.0437 |
| uncalibrated two-stage | -0.0166 | -2.12% | -0.0259 | -3.53% | -0.0087 | -0.97% | +0.0585 | -0.1050 | -0.0332 |

解读：

- 从最终最好结果看，`threshold-tuned two-stage` 的 macro-F1 仍低 `0.0126`，相对低 `1.71%`。
- 它的 `Enrolled acc` 比 final mixed ensemble 高 `0.0378`，但 Graduate acc 低 `0.0437`。
- 所以 two-stage tuned 的定位很清楚：更重视中间类召回，但不是总体最优。

## 面向报告的推荐排序

建议主报告按以下层次写：

1. **Final best model**：final OOF semester-members ensemble，macro-F1 0.7343。
2. **Best machine-only model**：machine-only OOF ensemble，macro-F1 0.7247。
3. **Best single classical model**：tuned CatBoost，macro-F1 0.7222。
4. **Ordinal/two-stage side-analysis**：threshold-tuned two-stage，macro-F1 0.7217，但 Enrolled acc 最高。
5. **Uncalibrated two-stage ablation**：macro-F1 0.7083，用来证明 naive two-stage 会压低中间类。

## 面向标签理解的结论

Two-stage 结果不支持把 `Enrolled` 简单合并到某一端。

更合适的解释是：

```text
Enrolled 是有序标签中的中间状态，但它不是一个容易通过硬阈值完全分开的纯净类别。
```

可视化、距离分析和 two-stage 实验共同指向同一结论：

- `Enrolled` 的确位于 `Dropout` 与 `Graduate` 之间。
- 但它内部有子群：dropout-side、enrolled-core、graduate-side、boundary。
- 直接 two-stage argmax 会让中间类被两端吞掉。
- OOF threshold tuning 可以救回一部分 Enrolled，但还不足以超过最强 one-stage ensemble。

## 可直接写进报告的结论

英文版：

```text
The calibrated two-stage ordinal decomposition improves substantially over the
naive two-stage recomposition, raising macro-F1 from 0.7083 to 0.7217 and
Enrolled recall from 0.4748 to 0.6176. However, it does not surpass the best
one-stage machine-only ensemble, which achieves 0.7247 macro-F1 and a higher
macro AUC. This suggests that the ordinal structure is useful for interpreting
the Enrolled class, but the best predictive strategy remains a calibrated
one-stage ensemble.
```

中文版：

```text
经过阈值校准后，二阶段有序分类明显优于未校准版本，macro-F1 从 0.7083
提升到 0.7217，Enrolled recall 从 0.4748 提升到 0.6176。但它仍未超过
最佳 one-stage machine-only ensemble，后者 macro-F1 为 0.7247 且 AUC 更高。
这说明有序二阶段结构对解释 Enrolled 类有价值，但最终预测上仍应优先采用
校准过的 one-stage ensemble。
```

## 结果来源

- `models/metrics/two_stage_threshold_tuned_metrics.csv`
- `models/metrics/two_stage_vs_one_stage_comparison.csv`
- `models/metrics/machine_only_ensemble_metrics.csv`
- `models/metrics/cross_seed_stacking_metrics.csv`
- `models/metrics/oof_probability_ensemble_semester_members_metrics.csv`
- `models/metrics/primary_contenders_summary.csv`
- `models/metrics/catboost_tuned_metrics.csv`
- `models/metrics/hist_column_sampling_metrics.csv`
