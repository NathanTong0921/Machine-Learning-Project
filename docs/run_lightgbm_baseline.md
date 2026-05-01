# run_lightgbm_baseline.py

## 作用

对应脚本：[scripts/run_lightgbm_baseline.py](/media/990Pro/ML/scripts/run_lightgbm_baseline.py)

这个脚本负责把 `LightGBM` 作为强表格 boosting baseline 接入当前项目。

## 为什么值得做

- `LightGBM` 是很强、很常见的表格 boosting 基线
- 训练速度快，参数搜索成本通常低于 CatBoost
- 适合和 `CatBoost / RF / TabPFN / Multi-View Gated MLP` 形成完整对照

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/lightgbm_metrics.csv`
- `models/metrics/lightgbm_metadata.json`
- `reports/figures/cm_lightgbm.png`

## 运行方式

```bash
python scripts/run_lightgbm_baseline.py
```

## 解释建议

- 如果它接近或超过 `CatBoost`，说明这份数据并不强依赖 ordered boosting。
- 如果它明显弱于 `CatBoost`，通常说明类别特征处理确实是关键。
