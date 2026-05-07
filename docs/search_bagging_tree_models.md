# search_bagging_tree_models.py

## 作用

对应脚本：`scripts/bagging/search_bagging_tree_models.py`

这个脚本系统比较三类 bagging-style 决策树模型：

- `bagging_decision_tree`：原生 Bagging，使用 bootstrap 抽样训练多棵决策树后聚合。
- `pasting_decision_tree`：Pasting-style 决策树集成，不放回抽样，即 `bootstrap=False`。
- `extra_trees`：Extremely Randomized Trees，分裂阈值也随机，比随机森林更随机。

说明：本项目目标是三分类，因此用户提到的 regression 类 bagging 没有直接作为回归任务实现；这里用 Pasting-style classification 覆盖“不放回抽样的 bagging”方向。

## Dataset Distribution Motivation

原论文 `docs/Predicting_Student_Dropout_and_Academic_Success.pdf` 的 Section 3.2.2 明确指出该任务存在类别不平衡：

| Target | Count | Share |
| --- | ---: | ---: |
| `Graduate` | 2209 / 4424 | 50% |
| `Dropout` | 1421 / 4424 | 32% |
| `Enrolled` | 794 / 4424 | 18% |

论文也提醒：高 accuracy 可能主要由 majority class `Graduate` 推动，从而牺牲 minority class `Enrolled` 的表现。因此这个 bagging 实验不只看 accuracy，而以 macro-F1 和 class-wise recall/F1 为主要解释指标。论文提到的相关方向包括 Balanced Random Forest、Easy Ensemble、Exactly/Roughly Balanced Bagging、Over-Bagging 和 SMOTE-Bagging；当前脚本先覆盖标准 Bagging、Pasting 和 Extra Trees，后续可以继续加 balanced bagging 变体。

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

如果输入不存在，先运行：

```bash
python scripts/preprocessing/prepare_data.py
```

## Grid 设置

选择指标：

- 5-fold CV `f1_macro`

Bagging / Pasting 搜索：

- `n_estimators`: `200`, `500`
- `max_samples`: `0.65`, `0.8`, `1.0`
- `max_features`: `0.7`, `1.0`
- base tree `max_depth`: `6`, `10`, `None`
- base tree `min_samples_leaf`: `1`, `3`, `5`

Extra Trees 搜索：

- `n_estimators`: `400`, `800`
- `max_depth`: `10`, `20`, `None`
- `min_samples_leaf`: `1`, `2`, `5`
- `max_features`: `"sqrt"`, `0.5`, `1.0`
- `class_weight`: `"balanced"`, `"balanced_subsample"`

## 输出

- `models/metrics/bagging_tree_grid.csv`
- `models/metrics/bagging_tree_metrics.csv`
- `models/metrics/bagging_tree_best_params.json`
- `reports/figures/bagging_tree_summary.png`
- `reports/figures/bagging_tree_confusion_grid.png`
- `reports/figures/cm_bagging_decision_tree.png`
- `reports/figures/cm_pasting_decision_tree.png`
- `reports/figures/cm_extra_trees.png`

## 当前结果

以下结果是在 Appendix 语义校对后重跑得到的版本：下游 pipeline 使用
`src.preprocessing.split_feature_columns()`，把 `Application mode`、`Course`、
`Previous qualification`、`Nacionality`、父母学历/职业和二元标志列作为
categorical 处理；`Application order` 保持 numeric/ordinal。

| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Macro AUC | CV Best Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pasting_decision_tree` | 0.7651 | 0.7113 | 0.7647 | 0.8864 | 0.7091 |
| `extra_trees` | 0.7568 | 0.7100 | 0.7631 | 0.8855 | 0.7121 |
| `bagging_decision_tree` | 0.7515 | 0.7079 | 0.7587 | 0.8853 | 0.7125 |

## 解读

1. 按 held-out test macro-F1 和 accuracy，`pasting_decision_tree` 最好，说明不放回抽样在这个数据集上略优于 bootstrap bagging。
2. `extra_trees` 的 CV macro-F1 最接近最高值，并且 held-out macro-F1 与 Pasting 只差约 `0.0013`，可以作为更随机 tree ensemble 的补充结果。
3. Minority class `Enrolled` 上，`bagging_decision_tree` 的 recall/F1 最高，分别为 `0.6092` 和 `0.5216`；`extra_trees` 次之，recall/F1 为 `0.6008` 和 `0.5191`；`pasting_decision_tree` 的 `Enrolled` F1 为 `0.5112`，但整体 accuracy 和 macro-F1 更好。
4. 这些模型没有超过 tuned CatBoost、LightGBM 或 OOF ensemble，但可以作为 open-ended exploration 中 bagging 家族的完整补充。

## 运行方式

完整 grid：

```bash
python scripts/bagging/search_bagging_tree_models.py --cv 5 --n-jobs -1
```

快速 smoke test：

```bash
python scripts/bagging/search_bagging_tree_models.py --quick --cv 2 --n-jobs 2
```
