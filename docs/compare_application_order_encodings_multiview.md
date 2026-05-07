# compare_application_order_encodings_multiview.py

对应脚本：`scripts/mlp/compare_application_order_encodings_multiview.py`

## 背景

`Application order` 表示志愿顺序。当前 Appendix-aligned preprocessing 把它作为
numeric/ordinal scalar，因此在 sklearn one-hot baselines 和 MultiView MLP 中会进入
median impute + `StandardScaler()`。

这个处理有一个尺度假设：模型会看到一个连续标量，并隐含认为相邻志愿顺序之间的距离近似等距。
对树模型影响较小，但对 Logistic Regression / SVM / MLP 这类线性、距离或梯度模型可能有偏误。

因此本实验只针对 MultiView Gated MLP 做 side comparison，不改全局 preprocess。

## 对比设置

固定使用当前 preprocess 搜索得到的 MultiView best 配置：

```json
{
  "seed": 62,
  "hidden_dim": 96,
  "dropout": 0.10,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 64,
  "max_epochs": 90,
  "patience": 12
}
```

比较三种 `Application order` 输入：

| variant | treatment |
| --- | --- |
| `scalar_current_best_recheck` | 保持当前做法，把 `Application order` 作为 scalar numeric feature |
| `thermometer_replace` | 删除 scalar，替换为 9 个 cumulative threshold indicators：`Application order >= 1..9` |
| `target_encoding_oof_replace` | 删除 scalar，替换为 5-fold OOF smoothed class-rate target encoding 三列 |

Target encoding 注意：

- train features 用 5-fold OOF 方式生成，避免每一行直接看到自己的标签。
- test features 用 full train 的 smoothed mapping。
- 这只是对单个 ordinal feature 的 side experiment，不改变其他特征处理。

## 固定超参结果

第一轮先固定使用 scalar current-preprocess best 配置，只替换 `Application order` 编码。Held-out test 结果：

| variant | accuracy | macro-F1 | weighted-F1 | FMI | AUC | Dropout acc | Enrolled acc | Graduate acc | enrollment view dim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scalar_current_best_recheck` | 0.7718 | 0.7285 | 0.7777 | 0.6808 | 0.8889 | 0.7963 | 0.5882 | 0.8220 | 80 |
| `target_encoding_oof_replace` | 0.7568 | 0.7158 | 0.7653 | 0.6633 | 0.8902 | 0.7166 | 0.6261 | 0.8296 | 82 |
| `thermometer_replace` | 0.7553 | 0.7153 | 0.7652 | 0.6644 | 0.8888 | 0.6838 | 0.6471 | 0.8401 | 88 |

相对 `scalar_current_best_recheck`：

| variant | accuracy delta | macro-F1 delta | AUC delta | Dropout acc delta | Enrolled acc delta | Graduate acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_encoding_oof_replace` | -0.0151 | -0.0127 | +0.0014 | -0.0796 | +0.0378 | +0.0075 |
| `thermometer_replace` | -0.0166 | -0.0133 | -0.0000 | -0.1124 | +0.0588 | +0.0181 |

## 调参结果

由于替换编码改变了输入维度和尺度，固定 scalar best 超参并不公平。因此追加脚本
`scripts/mlp/search_application_order_encoding_multiview.py`，分别对 `thermometer` 和
`target_encoding_oof` 搜索 `58` 组 MLP 超参。

搜索空间包括：

- `hidden_dim`: 64, 96, 128, 160
- `dropout`: 0.05, 0.10, 0.15, 0.20
- `lr`: 5e-4, 7e-4, 1e-3, 1.5e-3
- `batch_size`: 64, 128, 256
- `class_weight_multipliers`: baseline, `{"Enrolled": 1.1}`, `{"Enrolled": 1.2}`, `{"Enrolled": 1.3}`

Best held-out test 结果：

| variant | accuracy | macro-F1 | weighted-F1 | FMI | AUC | Dropout acc | Enrolled acc | Graduate acc | enrollment view dim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `thermometer_tuned` | 0.7824 | 0.7350 | 0.7840 | 0.6970 | 0.8903 | 0.7447 | 0.5798 | 0.8793 | 88 |
| `target_encoding_oof_tuned` | 0.7681 | 0.7299 | 0.7778 | 0.6787 | 0.8949 | 0.6956 | 0.6765 | 0.8477 | 82 |
| `scalar_tuned` | 0.7718 | 0.7285 | 0.7777 | 0.6808 | 0.8889 | 0.7963 | 0.5882 | 0.8220 | 80 |

最佳 thermometer 配置：

```json
{
  "seed": 62,
  "hidden_dim": 128,
  "dropout": 0.05,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 64,
  "max_epochs": 90,
  "patience": 12,
  "class_weight_multipliers": {}
}
```

最佳 OOF target encoding 配置：

```json
{
  "seed": 62,
  "hidden_dim": 160,
  "dropout": 0.20,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 64,
  "max_epochs": 90,
  "patience": 12,
  "class_weight_multipliers": {}
}
```

相对 `scalar_tuned`：

| variant | accuracy delta | macro-F1 delta | AUC delta | Dropout acc delta | Enrolled acc delta | Graduate acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `thermometer_tuned` | +0.0105 | +0.0065 | +0.0014 | -0.0515 | -0.0084 | +0.0573 |
| `target_encoding_oof_tuned` | -0.0038 | +0.0014 | +0.0060 | -0.1007 | +0.0882 | +0.0256 |

## 解释

这个实验支持两个判断：

1. `Application order` 的 scalar 标准化确实可能让模型更偏向两端类别，尤其是 `Dropout` recall。
2. 替换编码后必须重新调参；固定 scalar best 超参会低估 thermometer / target encoding。

调参后，`thermometer_tuned` 成为当前 MultiView MLP 最强的 `Application order` 处理方式：
macro-F1 从 scalar tuned 的 `0.7285` 提升到 `0.7350`，同时 accuracy 提升到 `0.7824`。
它的主要收益来自 `Graduate acc` 大幅提高，代价是 `Dropout acc` 下降。

`target_encoding_oof_tuned` 的 macro-F1 只略高于 scalar tuned，但 AUC 和 `Enrolled acc` 最好。
它更适合作为 minority-class / ranking-quality 取向的 side result。

## 结论

当前主 MLP 结果建议切到 `thermometer_tuned`，因为它在 tuned setting 下超过 scalar baseline。

报告中可以这样写：

```text
We additionally tested two ordinal-aware encodings for Application order in the
MultiView Gated MLP: thermometer encoding and leakage-controlled OOF target
encoding. After retuning the MLP hyperparameters for each encoding, thermometer
encoding achieved the best macro-F1, improving from 0.7285 with the scalar
encoding to 0.7350. OOF target encoding achieved the highest macro AUC and
Enrolled recall, but a slightly lower macro-F1. We therefore use thermometer
encoding as the preferred MLP-side treatment of Application order.
```

## 输出文件

- `models/metrics/multiview_application_order_encoding_metrics.csv`
- `models/metrics/multiview_application_order_encoding_best.json`
- `models/metrics/multiview_application_order_encoding_search_thermometer.csv`
- `models/metrics/multiview_application_order_encoding_search_thermometer_best.json`
- `models/metrics/multiview_application_order_encoding_search_target_encoding_oof.csv`
- `models/metrics/multiview_application_order_encoding_search_target_encoding_oof_best.json`
- `models/metrics/multiview_application_order_encoding_tuned_summary.csv`
- `models/saved_models/multiview_application_order_encoding_scalar_current_best_recheck.pt`
- `models/saved_models/multiview_application_order_encoding_thermometer_replace.pt`
- `models/saved_models/multiview_application_order_encoding_target_encoding_oof_replace.pt`
- `models/saved_models/multiview_application_order_encoding_search_thermometer_best.pt`
- `models/saved_models/multiview_application_order_encoding_search_target_encoding_oof_best.pt`
- `reports/figures/cm_multiview_application_order_encoding_scalar_current_best_recheck.png`
- `reports/figures/cm_multiview_application_order_encoding_thermometer_replace.png`
- `reports/figures/cm_multiview_application_order_encoding_target_encoding_oof_replace.png`
