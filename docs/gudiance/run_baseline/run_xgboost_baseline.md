# run_xgboost_baseline.py

## 作用

对应脚本：[scripts/boost/run_xgboost_baseline.py](/media/990Pro/ML/scripts/boost/run_xgboost_baseline.py)

这个脚本负责把 `XGBoost` 作为强表格 boosting baseline 接入当前项目。

## 为什么值得做

- `XGBoost` 是最经典、最稳的 boosting 对照之一
- 它能提供一个非常可靠的工程化 baseline
- 适合和 `CatBoost / LightGBM / RF / TabPFN / Multi-View Gated MLP` 一起比较

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/xgboost_metrics.csv`
- `models/metrics/xgboost_metadata.json`
- `reports/figures/cm_xgboost.png`

## 运行方式

```bash
python scripts/boost/run_xgboost_baseline.py
```

## 解释建议

- 如果它接近或超过 `CatBoost / LightGBM`，说明这份数据并不强依赖特定类别处理机制。
- 如果它弱于 `CatBoost`，那通常支持“类别特征友好性是关键”的判断。
