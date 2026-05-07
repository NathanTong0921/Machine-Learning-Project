# compare_trompt_style_models.py

## 作用

对应脚本：[scripts/transformer/compare_trompt_style_models.py](/media/990Pro/ML/scripts/transformer/compare_trompt_style_models.py)

这个脚本负责比较：

- 当前最优 `Multi-View Gated MLP`
- 调参后的 `FT-Transformer`
- 新实现的 `Trompt-style prompt transformer`
- 已经跑通的 `TabPFN`（若结果文件存在）

## 当前实现思路

这版 `Trompt-style` 不是论文逐字复刻，而是一个贴合当前项目结构的 prompt-based tabular transformer：

1. 类别特征做 embedding  
2. 数值特征做 periodic numerical tokenization  
3. 引入 learnable prompt tokens  
4. prompts 对 feature tokens 产生 sample-specific importance  
5. 用 prompts 聚合 feature tokens 后，再在 prompt 空间中做 transformer encoding  
6. 最后对 prompts 做池化并分类

## 为什么这样做

- 比继续放大标准 FT 更接近 prompt-style tabular modeling 的核心思路
- 能直接观察 prompt 聚合出来的 feature importance
- 更适合作为第三阶段的 fancy 模型尝试

## 输出

- `models/metrics/trompt_style_comparison.csv`
- `reports/figures/trompt_style_comparison.png`
- `reports/figures/trompt_style_confusion_grid.png`
- `reports/figures/cm_trompt_style_transformer.png`
- `models/saved_models/trompt_style_transformer.pt`
- `models/metrics/trompt_style_transformer_metadata.json`
- `models/metrics/trompt_style_feature_importance.csv`

## 运行方式

```bash
python scripts/transformer/compare_trompt_style_models.py
```

## 解释建议

- 如果它优于标准 FT，说明 prompt-based 路线对这份数据有价值。
- 如果它接近或超过 `Multi-View Gated MLP`，则说明 prompt-style transformer 值得继续打磨。
- 如果它没有明显提升，也不代表 prompt 路线失败，只说明当前 project-local 版本还不够强。

## Latest Recorded Run

- 日期：2026-04-22
- 对比结果：
  - `multiview_gated_mlp_best`
    - `accuracy = 0.7575`
    - `macro-F1 = 0.7217`
    - `AUC = 0.8925`
  - `ft_transformer_tuned`
    - `accuracy = 0.7515`
    - `macro-F1 = 0.7053`
    - `AUC = 0.8771`
  - `trompt_style_transformer`
    - `accuracy = 0.7312`
    - `macro-F1 = 0.6940`
    - `AUC = 0.8779`
  - `tabpfn`
    - `accuracy = 0.7861`
    - `macro-F1 = 0.7131`
    - `AUC = 0.9017`

- 当前结论：
  - 这版 `Trompt-style prompt transformer` 可以稳定训练
  - 但当前实现没有超过调过的 `FT-Transformer`
  - 它更像是一个可继续研究的 prompt 原型，而不是当前最强候选
