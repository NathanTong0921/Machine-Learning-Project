# search_hist_column_sampling.py

## 作用

对应脚本：`scripts/boost/search_hist_column_sampling.py`

这个脚本独立比较 XGBoost `tree_method="hist"` 和 LightGBM histogram boosting 在
`max_bin × column fraction` 下的表现。目标不是做大范围综合调参，而是回答：

- 默认 `max_bin=255/256` 是否必要；
- 每棵树使用多少比例的列更适合当前 36 维表格任务；
- `Enrolled` minority class 的 F1/recall 是否受 bucket 和列采样影响。

## 类别处理

实验使用 Appendix-aligned 字段语义，而不是 pandas dtype：

- categorical 列来自 `src.config.CATEGORICAL_COLUMNS`；
- `Application order` 保持 numeric/ordinal；
- LightGBM 使用原生 `categorical_feature`；
- XGBoost 参考 `run_xgboost_baseline.py`，把 categorical 列转为 pandas `category` 并启用 `enable_categorical=True`。

PDF Appendix A 主要给出 categorical code 的取值解释。当前 processed 数据共有 36 个特征，
其中 17 个 semantic categorical 特征和 19 个 numeric/ordinal 特征。`max_bin` 主要影响连续
或数值型 split candidate；名义编码列不应当按连续数值解释。

## Grid 设置

完整 grid：

- `max_bin`: `16`, `32`, `64`, `128`, `255`
- column fraction: `0.60`, `0.75`, `0.90`, `1.00`

column fraction 对 36 个输入特征的近似有效列数：

| Fraction | Approx. feature count |
| ---: | ---: |
| 0.60 | 22 |
| 0.75 | 27 |
| 0.90 | 33 |
| 1.00 | 36 |

XGBoost 只在主搜索中调 `colsample_bytree`，固定 `colsample_bylevel=1.0` 和
`colsample_bynode=1.0`，避免三个列采样参数累乘后每个 split 可用特征过少。
LightGBM 使用 sklearn API 的 `colsample_bytree`，即 per-tree feature fraction。

## 固定参数

LightGBM 固定在当前 tuned baseline 附近：

- `num_leaves=63`
- `learning_rate=0.03`
- `n_estimators=300`
- `min_child_samples=40`
- `subsample=0.8`
- `class_weight="balanced"`

XGBoost 固定在当前 baseline 附近：

- `tree_method="hist"`
- `max_depth=6`
- `learning_rate=0.05`
- `n_estimators=400`
- `subsample=0.9`
- `reg_alpha=0.0`
- `reg_lambda=1.0`

## 输出

- `models/metrics/hist_column_sampling_grid.csv`
- `models/metrics/hist_column_sampling_metrics.csv`
- `models/metrics/hist_column_sampling_best_params.json`

`grid.csv` 保存每组参数的 CV macro-F1、`Enrolled` F1/recall 和平均训练时间。
`metrics.csv` 保存每个模型按 CV macro-F1 选出的最佳配置在 held-out test 上的指标。
`best_params.json` 保存最佳配置、字段分组和测试指标。

## 当前结果

完整 5-fold grid 已运行，选择规则为 `cv_mean_f1_macro` 优先，`cv_mean_enrolled_f1` 作为 tie-break。

| Model | Best `max_bin` | Best column fraction | Approx. columns | CV macro-F1 | CV Enrolled F1 | Test macro-F1 | Test Enrolled F1 | Test Enrolled recall | Test AUC macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lightgbm` | 16 | 0.60 | 22 | 0.7174 | 0.5010 | 0.7196 | 0.5237 | 0.5336 | 0.8921 |
| `xgboost_hist` | 16 | 0.90 | 33 | 0.7076 | 0.4651 | 0.6642 | 0.3631 | 0.2479 | 0.8887 |

Top LightGBM CV candidates show that `max_bin=16` and `max_bin=64` are competitive with
or better than the default-style `255` setting on this split. The best XGBoost candidate also
uses `max_bin=16`; however, its held-out `Enrolled` recall is much lower than LightGBM's.

## 运行方式

快速 smoke test：

```bash
python scripts/boost/search_hist_column_sampling.py --quick --cv 2 --n-jobs 2
```

完整 5-fold grid：

```bash
python scripts/boost/search_hist_column_sampling.py --cv 5 --n-jobs 8
```
