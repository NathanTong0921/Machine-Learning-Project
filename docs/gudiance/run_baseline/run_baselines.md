# run_baselines.py

## 作用

对应脚本：[scripts/baseline/run_baselines.py](/media/990Pro/ML/scripts/baseline/run_baselines.py)

这个脚本负责完成课程项目中的监督学习 baseline，对训练集建模、在测试集评估，并保存指标与混淆矩阵。

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/classification_metrics.csv`
- `reports/figures/cm_logistic_regression.png`
- `reports/figures/cm_decision_tree.png`
- `reports/figures/cm_random_forest.png`
- `reports/figures/cm_gradient_boosting.png`
- `reports/figures/cm_adaboost.png`
- `reports/figures/cm_extra_trees.png`
- `reports/figures/cm_svm_rbf.png`

## 当前模型

- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `AdaBoostClassifier`
- `ExtraTreesClassifier`
- `SVC (RBF kernel)`

## 当前评估指标

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `f1_weighted`
- `roc_auc_ovr_macro`（当模型支持概率输出时）

## 运行方式

```bash
python scripts/baseline/run_baselines.py
```

## 记录建议

- 每次如果修改 baseline 模型集合、超参数或评价指标，都要更新本文件。
- 如果某个 baseline 明显失效，也应在这里记录原因，而不是只在代码里删除。
