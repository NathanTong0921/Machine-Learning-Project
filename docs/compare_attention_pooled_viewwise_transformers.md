# compare_attention_pooled_viewwise_transformers.py

## 作用

对应脚本：[scripts/transformer/compare_attention_pooled_viewwise_transformers.py](/media/990Pro/ML/scripts/transformer/compare_attention_pooled_viewwise_transformers.py)

这个脚本负责测试一个更强的 view pooling 选择：

- 标准 `FT-Transformer`
- `View-Wise Gated FT` with mean pooling
- `View-Wise Gated FT` with attention pooling
- 当前最优 `Multi-View Gated MLP`

## 新改动

### 之前

- 每个视图内的 token 用 mean pooling 聚合

### 现在

- 每个视图内引入 attention pooling
- 让模型自己学习视图内哪些 token 更重要

## 输出

- `models/metrics/attention_viewwise_transformer_comparison.csv`
- `reports/figures/attention_viewwise_transformer_comparison.png`
- `reports/figures/attention_viewwise_transformer_confusion_grid.png`
- `reports/figures/cm_attention_viewwise_gated_ft_transformer.png`
- `models/saved_models/attention_viewwise_gated_ft_transformer.pt`
- `models/metrics/attention_viewwise_gated_ft_transformer_metadata.json`
- `models/metrics/attention_viewwise_gated_ft_gate_summary.csv`

## 运行方式

```bash
python scripts/transformer/compare_attention_pooled_viewwise_transformers.py
```

## 解释建议

- 如果 attention pooling 优于 mean pooling，说明视图内 token 的重要性并不均匀。
- 如果它仍然没有超过 `Multi-View Gated MLP`，则更可能需要 hybrid 架构，而不是继续只在 Transformer 内部调 pooling。

## Latest Recorded Run

- 日期：2026-04-21
- 对比结果：
  - `multiview_gated_mlp_best`
    - `accuracy = 0.7575`
    - `macro-F1 = 0.7217`
    - `AUC = 0.8925`
  - `ft_transformer_tuned`
    - `accuracy = 0.7515`
    - `macro-F1 = 0.7053`
    - `AUC = 0.8771`
  - `viewwise_gated_ft_transformer`
    - `accuracy = 0.7417`
    - `macro-F1 = 0.7066`
    - `AUC = 0.8875`
  - `attention_viewwise_gated_ft_transformer`
    - `accuracy = 0.7440`
    - `macro-F1 = 0.7074`
    - `AUC = 0.8818`

- 当前结论：
  - attention pooling 相比 mean pooling 有小幅提升：
    - `accuracy`: `0.7417 -> 0.7440`
    - `macro-F1`: `0.7066 -> 0.7074`
  - 但提升幅度不大，且 `AUC` 反而低于 mean pooling 版本
  - 这说明“视图内 token 重要性不均匀”这一直觉是对的，但当前 attention pooling 还不足以显著改变整体格局
  - 当前最强模型仍然是 `Multi-View Gated MLP`
