# compare_advanced_models.py

## 作用

对应脚本：[scripts/compare_advanced_models.py](/media/990Pro/ML/scripts/compare_advanced_models.py)

这个脚本负责在同一套 `train/test` 数据划分下，正式比较：

- 当前最优 `Multi-View Gated MLP`
- 新实现的 `FT-Transformer`

它不仅保存指标，还会保存两者的混淆矩阵、模型权重和汇总图。

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/advanced_model_comparison.csv`
- `reports/figures/advanced_model_comparison.png`
- `reports/figures/cm_multiview_gated_mlp_best.png`
- `reports/figures/cm_ft_transformer.png`
- `reports/figures/advanced_model_confusion_grid.png`
- `models/saved_models/multiview_gated_mlp_best.pt`
- `models/saved_models/ft_transformer.pt`
- `models/metrics/multiview_gated_mlp_best_metadata.json`
- `models/metrics/ft_transformer_metadata.json`

## 当前比较协议

- 相同的数据划分
- 相同的主指标：
  - accuracy
  - precision_macro
  - recall_macro
  - f1_macro
  - f1_weighted
  - roc_auc_ovr_macro

## 当前模型配置

### Best Multi-View Gated MLP

- 来自前面 demo 搜索的最优配置
- `seed = 62`
- `hidden_dim = 64`
- `dropout = 0.10`

### FT-Transformer

- `seed = 62`
- `d_token = 48`
- `n_blocks = 2`
- `n_heads = 4`
- `dropout = 0.20`
- `lr = 0.0008`

## 运行方式

```bash
python scripts/compare_advanced_models.py
```

## 记录建议

- 若后续继续调 FT-Transformer，请把新配置写进本文件，不要只在代码里修改。
- 若 FT-Transformer 暂时不如 Multi-View Gated MLP，也要保留结果，不要删掉失败对比。

## Latest Recorded Run

- 日期：2026-04-21
- 对比结果：
  - `multiview_gated_mlp_best`
    - `accuracy = 0.7575`
    - `macro-F1 = 0.7217`
    - `AUC = 0.8925`
  - `ft_transformer`
    - `accuracy = 0.7515`
    - `macro-F1 = 0.7053`
    - `AUC = 0.8771`

- 当前结论：
  - 经过一轮小范围调参，`FT-Transformer` 已明显优于首版实现
  - 但在当前配置下，仍弱于当前最优 `Multi-View Gated MLP`
  - 两者的 accuracy 已较接近，但 `macro-F1` 仍有明显差距
  - 如果继续推进，优先调：
    - 更大的 `d_token`
    - 训练轮数 / early stopping
    - seed 稳定性
    - 轻度正则而不是更深层数
