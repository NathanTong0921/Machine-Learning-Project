# analyze_enrolled_feature_space_distance.py

## 作用

对应脚本：`scripts/reporting/analyze_enrolled_feature_space_distance.py`

这个 side-analysis 用来回答：`Enrolled` 样本在特征空间里更靠近 `Dropout` 端、`Graduate` 端，还是主要处在中间区域。它不训练新的主模型，也不覆盖已有脚本。

## 方法

使用和项目一致的语义类别列处理：

- 数值列：median imputation + standard scaling
- 类别列：most-frequent imputation + one-hot encoding

然后做两类距离估算。

### Endpoint Distance

只把 `Dropout` 和 `Graduate` 当作两端点：

- 计算 `Dropout` centroid
- 计算 `Graduate` centroid
- 对每个 `Enrolled` 样本计算到两端 centroid 的距离
- 计算 endpoint score：

```text
(distance_to_dropout - distance_to_graduate) / (distance_to_dropout + distance_to_graduate)
```

这个分数为正，表示更靠近 `Graduate`；为负，表示更靠近 `Dropout`。

同时用 endpoint-only kNN 估算局部邻居组成：

- `endpoint_knn_graduate_share`
- `endpoint_knn_dropout_share`

### Two-Stage Distance

对应前面的二阶段建模方式：

- Stage A：`Dropout` vs `NotDropout`
- Stage B：在非 Dropout 空间里做 `Enrolled` vs `Graduate`

这里训练两个 balanced logistic regression，只把它们当作可解释的线性边界，然后计算 signed distance：

- `stage_a_signed_distance > 0`：在 `NotDropout` 一侧
- `stage_b_signed_distance > 0`：在 `Graduate` 一侧
- `stage_b_signed_distance < 0`：在 `Enrolled` 一侧

## 输出

- `models/metrics/enrolled_feature_space_distances.csv`
- `models/metrics/enrolled_feature_space_distance_summary.csv`
- `models/metrics/enrolled_feature_space_distance_quantiles.csv`
- `models/metrics/enrolled_feature_space_distance_config.json`
- `reports/figures/enrolled_feature_space_distance_summary.png`
- `reports/figures/enrolled_feature_space_distance_scatter.png`

## 当前结果

正式运行命令：

```bash
python scripts/reporting/analyze_enrolled_feature_space_distance.py --neighbors 25 --score-band 0.05 --stage-band 0.10
```

### Endpoint Distance Tags

| tag | count | share | mean endpoint score | mean kNN graduate share | mean stage A dist | mean stage B dist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dropout_like | 57 | 0.0718 | -0.1200 | 0.1228 | -0.2953 | -0.9618 |
| graduate_like | 308 | 0.3879 | 0.0982 | 0.8652 | 0.2295 | -0.0566 |
| middle_or_ambiguous | 429 | 0.5403 | 0.0019 | 0.6367 | 0.0678 | -0.3509 |

Endpoint view 的结论：大部分 `Enrolled` 不是靠近 `Dropout` 端。只有约 7.2% 是明显 dropout-like；约 38.8% 已经 graduate-like；最多的是 54.0% 中间/模糊区。

### Two-Stage Distance Tags

| tag | count | share | mean endpoint score | mean kNN graduate share | mean stage A dist | mean stage B dist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dropout_side | 221 | 0.2783 | -0.0265 | 0.4766 | -0.1643 | -0.6079 |
| enrolled_core_after_not_dropout | 342 | 0.4307 | 0.0315 | 0.7073 | 0.1635 | -0.3015 |
| graduate_side_after_not_dropout | 71 | 0.0894 | 0.1002 | 0.8901 | 0.3303 | 0.2026 |
| stage_b_boundary | 160 | 0.2015 | 0.0761 | 0.8515 | 0.2492 | 0.0018 |

Two-stage view 的结论：有 27.8% 的真实 `Enrolled` 在 Stage A 线性边界上落到 `Dropout` 侧，这会让二阶段模型第一步就把这些样本截断成 Dropout。剩余样本中，43.1% 是较明确的 enrolled-core，20.2% 靠近 Enrolled/Graduate 边界，8.9% 已经在 Graduate 侧。

### 交叉表

| endpoint tag | dropout_side | enrolled_core | graduate_side | stage_b_boundary | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| dropout_like | 55 | 2 | 0 | 0 | 57 |
| graduate_like | 26 | 105 | 61 | 116 | 308 |
| middle_or_ambiguous | 140 | 235 | 10 | 44 | 429 |
| total | 221 | 342 | 71 | 160 | 794 |

这个交叉表说明：endpoint 上 graduate-like 的 `Enrolled` 并不一定应该直接改成 `Graduate`。其中只有 61/308 在二阶段 Stage B 上明确落到 Graduate 侧，116/308 位于 Stage B 边界，105/308 仍更像 enrolled-core。

## Tag 建议

推荐不要把 `Enrolled` 简单合并到 `Graduate` 或 `Dropout`，而是将其作为 ordinal middle state 保留，并在报告里解释为：

```text
Dropout < Enrolled/still active but not completed < Graduate
```

如果要进一步细分 `Enrolled`，建议使用四个 side-analysis tag：

- `dropout_side`：可能接近退学风险/失败端，二阶段第一步容易误杀。
- `enrolled_core_after_not_dropout`：最像真正的“正常学制结束时仍在读/未完成”。
- `stage_b_boundary`：介于 Enrolled 和 Graduate 之间，最可能需要 threshold calibration。
- `graduate_side_after_not_dropout`：最像延迟毕业/接近毕业端，但不能直接当作已毕业标签。

对建模的直接启示：

- 二阶段模型的 Stage A 需要保守一些，避免把真实 `Enrolled` 过早截断为 `Dropout`。
- Stage B 不应该用简单 argmax；应该对 `Enrolled` 做 decision bias 或 calibration。
- 主任务仍建议保留三分类标签，而不是硬改成两个二分类标签后直接拼回去。

## 运行方式

```bash
python scripts/reporting/analyze_enrolled_feature_space_distance.py --neighbors 25 --score-band 0.05 --stage-band 0.10
```
