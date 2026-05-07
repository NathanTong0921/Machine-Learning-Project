# search_classic_boosting_models.py

## 作用

对应脚本：`scripts/boost/search_classic_boosting_models.py`

这个脚本补齐经典 boosting 家族：

- `adaboost_stumps`：AdaBoost，分错样本在下一轮获得更高权重。
- `gbdt_log_loss`：sklearn `GradientBoostingClassifier`，即经典 GBM/GBDT；分类任务中使用 multinomial log-loss，逐轮拟合残差梯度。
- `hist_gradient_boosting`：sklearn histogram GBDT，更接近 LightGBM 的直方图加速思想，但仍是 sklearn 原生实现。

`XGBoost`、`LightGBM`、`CatBoost` 已经有单独脚本和 tuning 结果，不在这个脚本里重复运行。

## 类别处理

这个脚本沿用 Appendix-aligned preprocessing：用 `src.preprocessing.split_feature_columns()` 按字段语义拆分特征，而不是按 pandas dtype。

经典 sklearn boosting 不像 CatBoost/LightGBM 那样原生适合直接吃类别列，所以这里对 categorical 特征做 dense one-hot。`Application order` 保持 numeric/ordinal。

## Grid 设置

选择指标：

- 5-fold CV `f1_macro`

AdaBoost 搜索：

- base tree `max_depth`: `1`, `2`, `3`
- `n_estimators`: `100`, `200`, `400`
- `learning_rate`: `0.1`, `0.5`, `1.0`

当前没有运行 `LogisticRegression` 作为 AdaBoost base estimator。这里的 AdaBoost 指
`AdaBoostClassifier(estimator=DecisionTreeClassifier(...))`，也就是 shallow-tree/stump
路线。

GBDT/GBM 搜索：

- `n_estimators`: `100`, `200`, `400`
- `learning_rate`: `0.03`, `0.05`, `0.1`
- `max_depth`: `2`, `3`
- `subsample`: `0.8`, `1.0`

Histogram GBDT 搜索：

- `learning_rate`: `0.05`, `0.1`
- `max_iter`: `200`, `400`
- `max_leaf_nodes`: `15`, `31`
- `min_samples_leaf`: `10`, `20`
- `l2_regularization`: `0.0`, `0.1`
- `class_weight`: `"balanced"`

## 其他 Boosting 的边界

- `GBM` 和 `GBDT` 在这里视作同一类 sklearn `GradientBoostingClassifier`。
- `LogitBoost` 的核心思想是 logistic-loss additive boosting；当前用 `gbdt_log_loss` 覆盖可复现的 log-loss boosting 对照，但不声称是严格的原始 LogitBoost 实现。
- `AdaBoost + LogisticRegression base estimator` 没有运行；普通 `LogisticRegression` baseline、feature-engineered logistic 和 logistic stacking meta learner 已单独覆盖。
- `BrownBoost` 没有 sklearn 原生实现，若要做需要引入外部包或自写算法，不适合在课程项目里硬造不可验证 baseline。
- `RankBoost` 是排序算法，不是当前三分类任务的直接模型。

## 输出

- `models/metrics/classic_boosting_grid.csv`
- `models/metrics/classic_boosting_metrics.csv`
- `models/metrics/classic_boosting_best_params.json`
- `reports/figures/classic_boosting_summary.png`
- `reports/figures/classic_boosting_confusion_grid.png`
- `reports/figures/cm_adaboost_stumps.png`
- `reports/figures/cm_gbdt_log_loss.png`
- `reports/figures/cm_hist_gradient_boosting.png`

## 当前结果

完整 5-fold grid 已运行，选择指标是 CV macro-F1，held-out test 结果如下：

| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Macro AUC | CV Best Macro F1 | Enrolled Recall | Enrolled F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gbdt_log_loss` | 0.7515 | 0.7112 | 0.7592 | 0.8883 | 0.7216 | 0.6176 | 0.5307 |
| `hist_gradient_boosting` | 0.7560 | 0.7096 | 0.7611 | 0.8880 | 0.7201 | 0.5756 | 0.5180 |
| `adaboost_stumps` | 0.7402 | 0.7021 | 0.7517 | 0.8771 | 0.7082 | 0.6429 | 0.5195 |

解读：

1. `gbdt_log_loss` 是当前 classic boosting 里最强的单模型，macro-F1 与 Pasting/Extra Trees 接近。
2. `hist_gradient_boosting` 的 accuracy 略高，但 macro-F1 和 `Enrolled` F1 略低。
3. `adaboost_stumps` 的 `Enrolled` recall 最高，但整体 macro-F1 最低，说明它更偏向 minority recall，牺牲了整体平衡。
4. `GBM` 和 `GBDT` 在这里就是同一类 sklearn `GradientBoostingClassifier`；`LogitBoost` 只用 log-loss boosting 方向覆盖，不作为严格原始算法声明。

## 运行方式

完整 grid：

```bash
python scripts/boost/search_classic_boosting_models.py --cv 5 --n-jobs 32
```

快速 smoke test：

```bash
python scripts/boost/search_classic_boosting_models.py --quick --cv 2 --n-jobs 2
```
