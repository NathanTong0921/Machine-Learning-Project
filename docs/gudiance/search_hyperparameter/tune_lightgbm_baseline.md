# tune_lightgbm_baseline.py

## 作用

对应脚本：[scripts/boost/tune_lightgbm_baseline.py](/media/990Pro/ML/scripts/boost/tune_lightgbm_baseline.py)

这个脚本负责给 `LightGBM` 做一轮 `10-fold` 超参数搜索。

## 为什么值得做

- `LightGBM` 当前已经是很强的 baseline
- 训练速度快，适合做比 CatBoost 更大一点的搜索
- 很适合作为 tuned `CatBoost` 的直接对照

## 搜索协议

- `10-fold Stratified CV`
- 主评分指标：`macro-F1`
- 使用 categorical columns 的原生支持
- 带实时进度输出与状态文件

## 当前搜索范围

- `num_leaves`
- `max_depth`
- `learning_rate`
- `n_estimators`
- `min_child_samples`
- `subsample`
- `colsample_bytree`

## 输出

- `models/metrics/lightgbm_tuned_metrics.csv`
- `models/metrics/lightgbm_tuned_best_params.json`
- `models/metrics/lightgbm_tuned_cv_results.csv`
- `models/metrics/lightgbm_tuned_status.json`
- `reports/figures/cm_lightgbm_tuned.png`
- `reports/figures/lightgbm_tuned_summary.png`

## 运行方式

```bash
python scripts/boost/tune_lightgbm_baseline.py
```

如果要显式使用 108 个 CPU 线程：

```bash
LIGHTGBM_N_JOBS=108 python scripts/boost/tune_lightgbm_baseline.py
```

## 断点续跑

脚本会读取已有的：

```text
models/metrics/lightgbm_tuned_cv_results.csv
```

然后跳过已经完整写入 CSV 的参数组合，从下一个组合继续搜索。状态文件中如果停在某个组合的中间 fold，该组合会从头重跑一次，避免把不完整 fold 结果写进均值。

## 解释建议

- 如果 tuned LightGBM 接近或超过 tuned CatBoost，就说明这份数据未必强依赖 CatBoost 的 ordered boosting 机制。
- 如果 tuned LightGBM 仍低于 tuned CatBoost，则 CatBoost 更可能是当前最佳传统表格模型。
