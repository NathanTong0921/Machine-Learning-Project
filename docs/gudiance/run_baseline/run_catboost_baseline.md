# run_catboost_baseline.py

## 作用

对应脚本：[scripts/boost/run_catboost_baseline.py](/media/990Pro/ML/scripts/boost/run_catboost_baseline.py)

这个脚本负责把 `CatBoost` 作为最优先的强表格 baseline 接入当前项目。

## 为什么优先

- 当前数据中有很多整数列本质上是类别特征
- CatBoost 原生支持类别特征
- 对中小规模表格分类通常非常稳

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/catboost_metrics.csv`
- `models/metrics/catboost_metadata.json`
- `reports/figures/cm_catboost.png`

## 运行方式

```bash
python scripts/boost/run_catboost_baseline.py
```

## 解释建议

- 它应该优先和 `RandomForest`、`soft voting`、`Multi-View Gated MLP`、`TabPFN` 做对比。
- 如果它的 `macro-F1` 很强，就说明当前这份数据非常适合类别友好的 boosting。
