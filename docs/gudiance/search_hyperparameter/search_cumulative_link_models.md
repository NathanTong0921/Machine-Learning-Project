# search_cumulative_link_models.py

## 作用

对应脚本：`scripts/ensemble/search_cumulative_link_models.py`

这个实验把 `Dropout < Enrolled < Graduate` 当作有序目标，用两个递进二分类器近似 cumulative link / ordinal decomposition：

- Stage A：`Dropout` vs `NotDropout`
- Stage B：只在非 Dropout 训练样本中学习 `Enrolled` vs `Graduate`

最后用两个二分类概率还原三分类概率：

- `P(Dropout)=1-P(NotDropout)`
- `P(Enrolled)=P(NotDropout)*(1-P(Graduate|NotDropout))`
- `P(Graduate)=P(NotDropout)*P(Graduate|NotDropout)`

## 当前成员

- `catboost_cumulative`
- `lightgbm_hist_cumulative`
- `gbdt_cumulative`
- `cumulative_link_ensemble`

这些都是追加实验，不覆盖原始 multiclass 脚本或已有结果。

## 输出

- `models/metrics/cumulative_link_metrics.csv`
- `models/metrics/cumulative_link_binary_stage_metrics.csv`
- `models/metrics/cumulative_link_weight_grid.csv`
- `models/metrics/cumulative_link_best_weights.json`
- `reports/figures/cumulative_link_summary.png`

## 当前结果

正式运行命令：

```bash
python scripts/ensemble/search_cumulative_link_models.py --cv 5 --n-jobs 96
```

测试集三分类结果：

| model | accuracy | macro-F1 | weighted-F1 | FMI | macro AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| catboost_cumulative_test | 0.7628 | 0.7111 | 0.7649 | 0.6729 | 0.8896 | 0.8080 | 0.5126 | 0.8235 |
| gbdt_cumulative_test | 0.7605 | 0.7103 | 0.7626 | 0.6676 | 0.8791 | 0.8056 | 0.5168 | 0.8190 |
| lightgbm_hist_cumulative_test | 0.7651 | 0.7078 | 0.7647 | 0.6811 | 0.8901 | 0.7822 | 0.4832 | 0.8552 |
| cumulative_link_ensemble_test | 0.7658 | 0.7083 | 0.7648 | 0.6803 | 0.8925 | 0.8056 | 0.4748 | 0.8446 |

OOF 权重：

| member | weight |
| --- | ---: |
| catboost_cumulative | 0.30 |
| lightgbm_hist_cumulative | 0.40 |
| gbdt_cumulative | 0.30 |

二阶段本身并不差：

| stage | best member | macro-F1 | AUC |
| --- | --- | ---: | ---: |
| Dropout vs NotDropout | catboost_cumulative | 0.8512 | 0.9167 |
| Enrolled vs Graduate | catboost_cumulative | 0.8018 | 0.9069 |

解释：

- 这个 decomposition 合理，但当前数据上不如直接 multiclass/OOF ensemble。`cumulative_link_ensemble_test` macro-F1 为 0.7083，低于 machine-only OOF ensemble 的 0.7247，也低于最终含高级成员 ensemble 的约 0.7343。
- 问题主要出在概率重组后 `Enrolled` 被压低：ensemble 的 `enrolled_accuracy` 只有 0.4748。单独看 Stage B 的 `Enrolled vs Graduate` AUC 很高，说明二分类可分；但乘上 `P(NotDropout)` 并用 argmax 三分类后，中间类容易被两侧类吃掉。
- 如果继续挖这个方向，下一步应该调三分类决策规则，而不是再盲目加大二分类模型。例如给 `Enrolled` 增加 calibration / threshold bonus，或在重组概率后单独搜索 `Enrolled` decision bias。

## 运行方式

```bash
python scripts/ensemble/search_cumulative_link_models.py --cv 5 --n-jobs 96
```

快速 smoke test：

```bash
python scripts/ensemble/search_cumulative_link_models.py --quick --cv 2 --n-jobs 4
```
