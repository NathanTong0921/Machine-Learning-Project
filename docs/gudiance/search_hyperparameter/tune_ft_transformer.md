# tune_ft_transformer.py

## 作用

对应脚本：[scripts/transformer/tune_ft_transformer.py](/media/990Pro/ML/scripts/transformer/tune_ft_transformer.py)

这个脚本负责给 `FT-Transformer` 做一轮小范围调参，用统一 split 和统一指标筛出当前最优 FT 配置。

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/ft_tuning_results.csv`
- `models/metrics/ft_tuning_ranked.md`
- `models/metrics/ft_tuning_best.json`
- `reports/figures/ft_tuning_table.png`

## 当前搜索范围

- `d_token`
- `n_blocks`
- `attention_dropout`
- `ff_dropout`
- `learning rate`
- 训练轮数和 early stopping patience

这是有意控制过的小搜索，不是大规模超参数优化。

## 运行方式

```bash
python scripts/transformer/tune_ft_transformer.py
```

## 选择规则

- 优先看 `macro-F1`
- 再看 `roc_auc_ovr_macro`
- 最后才看 `accuracy`

## 记录建议

- 若后续扩大搜索范围，要在本文件写清楚哪些超参数被新增。
- 若把最优 FT 配置同步到 `compare_advanced_models.py`，也应在这里记录更新时间和结果变化。

## Latest Recorded Run

- 日期：2026-04-21
- 第二轮搜索扩大了 `d_token` 到 `64 / 80`
- 同时把训练预算从 `max_epochs=60, patience=10` 拉长到：
  - `max_epochs = 90`
  - `patience = 16`

- 第二轮最佳 trial 仍然没有超过上一轮：
  - 当前轮最佳：`ft_wider_tokens`
  - `d_token = 48`
  - `macro-F1 = 0.7033`
  - `accuracy = 0.7402`
  - `AUC = 0.8837`

- 当前结论：
  - 更大的 `d_token`（`64 / 80`）没有继续带来收益
  - 更长训练和更大 patience 也没有释放出更强的 FT 表现
  - 对这份数据来说，FT 当前更可能受模型形式与数据规模匹配限制，而不只是训练不够久
  - 因此正式对比脚本暂时保留上一轮更优的 FT 配置，不同步这轮结果
