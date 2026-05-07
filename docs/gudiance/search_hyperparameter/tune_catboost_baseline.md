# tune_catboost_baseline.py

## 作用

对应脚本：[scripts/boost/tune_catboost_baseline.py](/media/990Pro/ML/scripts/boost/tune_catboost_baseline.py)

这个脚本负责给 `CatBoost` 做一轮 `10-fold` 超参数搜索。

## 为什么值得做

- CatBoost 当前已经是非常强的 baseline
- 它最贴这份数据的“类别特征很多”这一结构
- 因此很值得单独做一轮更正式的调参，而不是只跑默认/单点配置

## 搜索协议

- `10-fold Stratified CV`
- 主评分指标：`macro-F1`
- 原生 `cat_features` 传入

## 当前搜索范围

- `depth`
- `learning_rate`
- `iterations`
- `l2_leaf_reg`
- `random_strength`

## 输出

- `models/metrics/catboost_tuned_metrics.csv`
- `models/metrics/catboost_tuned_best_params.json`
- `reports/figures/cm_catboost_tuned.png`
- `reports/figures/catboost_tuned_summary.png`

## 运行方式

```bash
python scripts/boost/tune_catboost_baseline.py
```

## 解释建议

- 如果 tuned CatBoost 超过当前 `Multi-View Gated MLP` 的 `macro-F1`，那它就会成为最值得重视的传统表格主力模型。
- 即使没有超过，它也能给出一个非常强的表格 boosting 参考线。
