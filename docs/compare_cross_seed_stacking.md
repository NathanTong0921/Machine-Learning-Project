# compare_cross_seed_stacking.py

## 作用

对应脚本：`scripts/ensemble/compare_cross_seed_stacking.py`

这个脚本用于检查 cross-seed stacking 是否比简单 cross-seed soft average 更有价值。当前默认只使用传统机器学习成员：

- CatBoost tuned
- LightGBM hist-column best
- GBDT log-loss

明确排除 MLP、Transformer、TabPFN 和其他深度学习成员。

## 方法

对每个 seed 和每个成员：

- 用 stratified K-fold 生成训练集 OOF 概率；
- 用全训练集拟合一次，生成测试集概率；
- 把所有 `seed × member × class` 的 OOF 概率拼成二层特征；
- 用 `LogisticRegression(max_iter=5000, class_weight="balanced")` 作为 stacking meta learner。

同时保存两个对照：

- `cross_seed_soft_average`：所有 seed/member 概率直接平均；
- `cross_seed_stacking`：OOF 概率训练 meta learner 后预测。

## 输出

- `models/metrics/cross_seed_stacking_metrics.csv`
- `models/metrics/cross_seed_stacking_config.json`
- `reports/figures/cross_seed_stacking_summary.png`

## 当前结果

正式运行命令：

```bash
python scripts/ensemble/compare_cross_seed_stacking.py --cv 5 --n-jobs 96 --seeds 42 62 82
```

测试集主要结果：

| model | accuracy | macro-F1 | weighted-F1 | FMI | macro AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cross_seed_soft_average_test | 0.7673 | 0.7247 | 0.7727 | 0.6747 | 0.8976 | 0.7330 | 0.6092 | 0.8462 |
| cross_seed_stacking_test | 0.7508 | 0.7172 | 0.7633 | 0.6498 | 0.8920 | 0.7283 | 0.6765 | 0.7919 |

单 seed 的 soft average：

| model | accuracy | macro-F1 | macro AUC | enrolled acc |
| --- | ---: | ---: | ---: | ---: |
| soft_average_seed42_test | 0.7711 | 0.7283 | 0.8979 | 0.6092 |
| soft_average_seed62_test | 0.7696 | 0.7286 | 0.8972 | 0.6218 |
| soft_average_seed82_test | 0.7658 | 0.7241 | 0.8970 | 0.6218 |

解释：

- Cross-seed soft average 是稳定增强，但没有超过这次 machine-only OOF 权重 ensemble：两者 test macro-F1 基本一样，分别是 0.7247 和 0.7247；machine-only OOF 权重 ensemble 的 AUC 略高，0.8982 vs 0.8976。
- Cross-seed stacking 提升了 Enrolled accuracy（0.6765），但整体 macro-F1 降到 0.7172，说明二层 LR 过度偏向难类召回，牺牲了 Dropout/Graduate 的平衡。
- 当前报告里更建议采用 `machine_only_ensemble_test` 或 `cross_seed_soft_average_test`，不建议把 `cross_seed_stacking_test` 作为主结果。

## 运行方式

```bash
python scripts/ensemble/compare_cross_seed_stacking.py --cv 5 --n-jobs 96 --seeds 42 62 82
```
