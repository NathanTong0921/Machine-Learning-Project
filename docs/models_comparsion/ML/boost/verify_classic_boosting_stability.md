# verify_classic_boosting_stability.py

## 作用

对应脚本：`scripts/boost/verify_classic_boosting_stability.py`

这个脚本用于 classic boosting 的 cross-set / cross-seed verification。它读取 `models/metrics/classic_boosting_best_params.json` 中 full grid 选出的最佳参数，然后对多个 seed 重新做 stratified train/test split。

每个 seed 同时影响：

- train/test split
- 模型 `random_state`

因此结果不只验证“换随机初始化”，也验证“换 held-out set” 后是否稳定。

## 默认设置

- seeds: `42`, `62`, `82`, `102`, `122`
- test size: `0.30`
- metric: macro-F1 first，同时记录 accuracy、macro AUC 和 class-wise F1

## 输出

- `models/metrics/classic_boosting_cross_seed_metrics.csv`
- `models/metrics/classic_boosting_cross_seed_summary.csv`
- `models/metrics/classic_boosting_cross_seed_summary.json`
- `reports/figures/classic_boosting_cross_seed_summary.png`

## 当前结果

默认 5 个 seed 已运行：`42`, `62`, `82`, `102`, `122`。每个 seed 都重新做 stratified train/test split，并同步作为模型 `random_state`。

| Model | Mean Accuracy | Mean Macro F1 | Macro F1 Std | Mean Macro AUC | Mean Enrolled F1 | Enrolled F1 Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gbdt_log_loss` | 0.7664 | 0.7237 | 0.0154 | 0.8891 | 0.5360 | 0.0180 |
| `hist_gradient_boosting` | 0.7706 | 0.7235 | 0.0109 | 0.8912 | 0.5302 | 0.0153 |
| `adaboost_stumps` | 0.7530 | 0.7155 | 0.0151 | 0.8728 | 0.5323 | 0.0215 |

结论：cross-set / cross-seed 下，`gbdt_log_loss` 和 `hist_gradient_boosting` 的 mean macro-F1 基本持平；`hist_gradient_boosting` accuracy/AUC 稍高且方差更小，`gbdt_log_loss` 的 `Enrolled` F1 略高。AdaBoost 仍然整体偏弱。

## 运行方式

```bash
python scripts/boost/verify_classic_boosting_stability.py
```

自定义 seed：

```bash
python scripts/boost/verify_classic_boosting_stability.py --seeds 42 62 82 102 122 142 162
```
