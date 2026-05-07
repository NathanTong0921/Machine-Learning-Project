# 超参搜索与实验跑法（简版）

用途：把 `gudiance/search_hyperparameter/` 下的实验按推荐顺序快速跑通，先把关键 baseline 收口，再做 MoE 与集成对比。

## 推荐顺序（最短路径）

1. **先跑基线调参**
   - `tune_catboost_baseline.py`（10-fold CV，macro-F1）
   - `tune_lightgbm_baseline.py`（10-fold CV，macro-F1）

2. **再补齐经典 boosting 对照**
   - `search_classic_boosting_models.py`
   - 关注 `gbdt_log_loss`（通常是该组最强）

3. **再做 Transformer 轻量调参**
   - `tune_ft_transformer.py`
   - 以 `macro-F1` 为第一优先

4. **若 preprocessing 有变更，必须重搜核心 Deep 模型**
   - `search_multiview_current_preprocess.py`
   - 重点：App order 从 one-hot 改为 numeric 后旧超参不能直接复用

5. **补充有序分解试验（可选）**
   - `search_cumulative_link_models.py`
   - 用于诊断 `Enrolled/Graduate` 软边界，不做主线优先选择

6. **MoE 路线**
   - `search_tabular_moe.py`（无监督 GMM 路线，负向基线）
   - `search_supervised_error_router_moe.py`（监督路由 MoE，当前最有价值）
   - `search_supervised_error_router_moe_v2.md`（关注跟进脚本）

7. **纯机器模型集成（排除 DL）**
   - `tune_machine_only_ensemble.py`

## 常用启动命令（简化）

```bash
python scripts/boost/tune_catboost_baseline.py
python scripts/boost/tune_lightgbm_baseline.py

python scripts/boost/search_classic_boosting_models.py --cv 5 --n-jobs 32
python scripts/transformer/tune_ft_transformer.py

# 两卡并行重搜（多视图当前 preprocess）
env HIP_VISIBLE_DEVICES=0 python scripts/mlp/search_multiview_current_preprocess.py --shard-index 0 --num-shards 2
env HIP_VISIBLE_DEVICES=1 python scripts/mlp/search_multiview_current_preprocess.py --shard-index 1 --num-shards 2

python scripts/ensemble/search_cumulative_link_models.py --cv 5 --n-jobs 96
python scripts/ensemble/search_tabular_moe.py
python scripts/ensemble/search_supervised_error_router_moe.py --cv 5 --n-jobs 64
python scripts/ensemble/search_supervised_error_router_moe.py --quick --cv 5 --n-jobs 64 --output-prefix supervised_error_router_moe_core3
python scripts/ensemble/tune_machine_only_ensemble.py --cv 5 --n-jobs 96 --max-ensemble-members 6
```

## 结果优先级（报告角度）

- 优先：`tune_catboost_baseline`、`tune_lightgbm_baseline`、`tune_machine_only_ensemble`、`search_supervised_error_router_moe`
- 次优：`search_classic_boosting_models`、`tune_ft_transformer`、`search_cumulative_link_models`
- 诊断/对照：`search_tabular_moe`

## 记忆版结论

- 这套流程是“先稳树模型，再补 DL，再上 MoE，再上纯 ML ensemble”，最后再决定是否把 DL 分支并入最终混合。
