# tune_machine_only_ensemble.py

## 作用

对应脚本：`scripts/ensemble/tune_machine_only_ensemble.py`

这个脚本只调传统机器学习 ensemble，明确排除：

- MLP / Multi-View Gated MLP
- FT-Transformer / View-wise Transformer / Trompt / Piecewise Transformer
- TabPFN

允许成员只来自 bagging、Extra Trees、classic boosting、CatBoost、LightGBM、XGBoost hist。

## 指标

输出包含：

- accuracy
- FMI / Fowlkes-Mallows index
- macro-F1 / weighted-F1
- macro AUC
- `dropout_accuracy`
- `enrolled_accuracy`
- `graduate_accuracy`

这里 per-class accuracy 定义为 `TP / true class count`，等价于 class recall。

## 输出

- `models/metrics/machine_only_ensemble_metrics.csv`
- `models/metrics/machine_only_ensemble_member_oof_scores.csv`
- `models/metrics/machine_only_ensemble_weight_grid.csv`
- `models/metrics/machine_only_ensemble_best_weights.json`
- `reports/figures/machine_only_ensemble_summary.png`

## 当前结果

最终运行命令使用 5-fold OOF 和 `--n-jobs 96`：

```bash
python scripts/ensemble/tune_machine_only_ensemble.py --cv 5 --n-jobs 96 --max-ensemble-members 6
```

Machine-only ensemble 的测试集结果：

| model | accuracy | macro-F1 | weighted-F1 | FMI | macro AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| machine_only_ensemble_test | 0.7696 | 0.7247 | 0.7732 | 0.6784 | 0.8982 | 0.7330 | 0.5924 | 0.8567 |

OOF 训练集结果：

| model | accuracy | macro-F1 | weighted-F1 | FMI | macro AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| machine_only_ensemble_oof | 0.7868 | 0.7404 | 0.7881 | 0.7006 | 0.8939 | 0.7636 | 0.5755 | 0.8777 |

最终权重：

| member | weight |
| --- | ---: |
| CatBoost tuned | 0.30 |
| LightGBM hist-column best | 0.30 |
| GBDT log-loss | 0.10 |
| HistGradientBoosting | 0.15 |
| AdaBoost stumps | 0.10 |
| Extra Trees | 0.05 |
| LightGBM tuned | 0.00 |
| Pasting Decision Tree | 0.00 |
| Bagging Decision Tree | 0.00 |
| XGBoost hist-column best | 0.00 |

解释：

- 相比最强 machine-only 单模型 `lightgbm_hist_column_best_test`，macro-F1 从 0.7196 提升到 0.7247，macro AUC 从 0.8921 提升到 0.8982。
- 相比含 TabPFN / MLP / semester tree 的最终大 ensemble，仍低一些：最终大 ensemble macro-F1 约 0.7343，macro AUC 约 0.9013。
- XGBoost hist-column 的 FMI 很高，但主要来自 Dropout/Graduate 倾斜，Enrolled accuracy 只有 0.2479，所以权重搜索把它降为 0。
- Cross-seed soft average 的 test macro-F1 也是 0.7247，但 AUC 略低于本脚本的 OOF 权重 ensemble；cross-seed stacking 则降到 0.7172，不建议作为主结果。

## 运行方式

```bash
python scripts/ensemble/tune_machine_only_ensemble.py --cv 5 --n-jobs 96 --max-ensemble-members 6
```

快速检查：

```bash
python scripts/ensemble/tune_machine_only_ensemble.py --quick --cv 2 --n-jobs 4
```
