# search_and_ensemble_baselines.py

## 作用

对应脚本：[scripts/ensemble/search_and_ensemble_baselines.py](/media/990Pro/ML/scripts/ensemble/search_and_ensemble_baselines.py)

这个脚本把两条课程主线合并在一起：

1. model selection / hyperparameter search  
2. ensemble learning

## 覆盖模型

- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `AdaBoostClassifier`
- `ExtraTreesClassifier`
- `SVC`

## 搜索方式

- 使用 `GridSearchCV`
- `10-fold Stratified CV`
- 主评分指标：`f1_macro`

## Voting 权重

- `Voting` 不再使用等权
- 权重来自训练集上的 `10-fold out-of-fold` 表现
- 当前实现用：
  - `10-fold macro-F1` 作为原始权重
  - 再归一化为 voting 权重

## 输出

### 单模型搜索

- `models/metrics/baseline_gridsearch_metrics.csv`
- `models/metrics/baseline_gridsearch_best_params.csv`
- `reports/figures/baseline_gridsearch_summary.png`

### 集成模型

- `models/metrics/*_10fold_weights.csv`
- `reports/figures/*_loocv_weights.png`
- `models/metrics/baseline_ensemble_metrics.csv`
- `reports/figures/baseline_ensemble_summary.png`
- `reports/figures/baseline_ensemble_confusion_grid.png`

## 当前集成方式

- `voting_soft`
- `voting_hard`
- `stacking`

## 运行方式

```bash
python scripts/ensemble/search_and_ensemble_baselines.py
```

## 解释建议

- 先看单模型最优结果，再看 ensemble 是否真的带来增益。
- 再看 `10-fold` 学出来的权重是否合理，例如强模型是否得到更大权重。
- 如果 stacking 明显优于单模型，说明不同模型之间有互补性。
- 如果 soft voting 不如最佳单模型，也很正常，这说明模型相关性较高。
