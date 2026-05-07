# run_feature_engineered_baselines.py

## 作用

对应脚本：[scripts/feature_engineering/run_feature_engineered_baselines.py](/media/990Pro/ML/scripts/feature_engineering/run_feature_engineered_baselines.py)

这个脚本负责补一条课程材料非常自然的 baseline 线：

- 仍然使用简单分类器
- 但显式加入特征工程

## 为什么值得做

课程材料里明确涉及：

- polynomial features
- feature engineering
- model selection
- regularization

这意味着 baseline 不一定只能是“原始特征 + 模型”，也可以是：

- 特征工程 + 基础模型

## 当前模型

### 1. logistic_poly2

- 数值特征：
  - 缺失值填充
  - 二阶 `PolynomialFeatures`
  - 标准化
- 类别特征：
  - one-hot
- 分类器：
  - `LogisticRegression`

### 2. logistic_rbf_nystroem

- 先做标准数值/类别预处理
- 再做 `Nystroem` 近似 RBF feature map
- 最后接 `LogisticRegression`

## 输出

- `models/metrics/feature_engineered_baseline_metrics.csv`
- `reports/figures/cm_logistic_poly2.png`
- `reports/figures/cm_logistic_rbf_nystroem.png`

## 运行方式

```bash
python scripts/feature_engineering/run_feature_engineered_baselines.py
```

## 解释建议

- 如果它们优于 plain logistic regression，说明特征工程确实有价值。
- 如果它们接近树模型，说明“简单模型 + 好特征”对这份数据是有效路线。
- 如果它们仍不如高级模型，可以把它们写成课程式强化 baseline。
