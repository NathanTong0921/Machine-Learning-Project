# Baseline 快速执行说明（简版）

用途：把 `run_baseline/` 下的基线实验按最小代价跑通，先拿到可比对结果，再补充 advanced 实验。

## 推荐顺序（最低风险）

1. **先跑最小经典基线**
   - `run_baselines.py`（逻辑回归、树模型、AdaBoost 等）
2. **补不平衡基线**
   - `run_balanced_rf_baseline.py`（重点看 `macro-F1` 改善）
3. **补强树模型家族**
   - `run_catboost_baseline.py`
   - `run_xgboost_baseline.py`
   - `run_lightgbm_baseline.py`
4. **补高级/拓展线**
   - `run_feature_engineered_baselines.py`（简单模型 + 特征工程）
   - `run_tabpfn_baseline.py`（外部 fancy baseline）
   - `run_advanced.py`（Multi-View MLP 主线）
5. **补实验性聚类**
   - `run_clustering.py`（用于数据结构观察，不作为分类主线上限对标）

## 一键执行命令（按顺序）

```bash
python scripts/baseline/run_baselines.py
python scripts/bagging/run_balanced_rf_baseline.py
python scripts/boost/run_catboost_baseline.py
python scripts/boost/run_xgboost_baseline.py
python scripts/boost/run_lightgbm_baseline.py
python scripts/feature_engineering/run_feature_engineered_baselines.py
python scripts/transformer/run_tabpfn_baseline.py
python scripts/mlp/run_advanced.py
python scripts/clustering/run_clustering.py
```

## 产物速查（方便归档）

- `models/metrics/classification_metrics.csv`
- `models/metrics/balanced_rf_metrics.csv`
- `models/metrics/catboost_metrics.csv`
- `models/metrics/xgboost_metrics.csv`
- `models/metrics/lightgbm_metrics.csv`
- `models/metrics/feature_engineered_baseline_metrics.csv`
- `models/metrics/tabpfn_metrics.csv`
- `models/metrics/advanced_metrics.csv`
- `models/metrics/clustering_metrics.csv`

对应混淆矩阵通常位于 `reports/figures/`，每个脚本自带文件名。

## 结果判读优先级（你可以直接贴报告）

1. 先看 `macro-F1`，再看 `accuracy`/`AUC`。
2. 其次关注 `Enrolled` 类别指标（该类是项目中的弱点）。
3. 若某脚本明显退化，先从 preprocessing、类别列声明、class weight 三处检查，不要先改模型。
4. 在 final 对比中保留“可复现”的版本号（脚本 + 数据切分 + 关键超参）作为主线证据。

## 快速结论模板（记录在日志）

- baseline 版本：`run_baselines.py`（时间，硬件，提交哈希）
- 核心结论：哪个模型最强、是否有少数类受益、是否有训练不稳定现象
- 决策：是否需要重跑、是否进到 tune/search/ensemble 阶段
