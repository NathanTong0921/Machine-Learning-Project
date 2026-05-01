# compare_viewwise_gated_transformers.py

## 作用

对应脚本：[scripts/compare_viewwise_gated_transformers.py](/media/990Pro/ML/scripts/compare_viewwise_gated_transformers.py)

这个脚本负责正式比较：

- 当前最优 `Multi-View Gated MLP`
- 调参后的标准 `FT-Transformer`
- 新的 `View-Wise Gated FT-Transformer`

## 新变体的核心

### Step 1: token encoding

- 类别特征继续做 embedding
- 数值特征使用 periodic numerical tokenizer

### Step 2: transformer encoding

- 所有 token 先经过同一个 Transformer backbone

### Step 3: view-wise token pooling

- 按数据的天然视图分组：
  - 入学背景
  - 家庭财务
  - 第一学期
  - 第二学期与宏观变量
- 对每个视图内的 token 做 pooling，得到 view-level representation

### Step 4: gated fusion

- 用 gating network 学习 4 个视图如何加权融合
- 最终分类头使用：
  - `CLS`
  - 所有 view pooled 向量
  - gated fused 向量
  - gate 权重

## 输出

- `models/metrics/viewwise_gated_transformer_comparison.csv`
- `reports/figures/viewwise_gated_transformer_comparison.png`
- `reports/figures/viewwise_gated_transformer_confusion_grid.png`
- `reports/figures/cm_viewwise_gated_ft_transformer.png`
- `models/saved_models/viewwise_gated_ft_transformer.pt`
- `models/metrics/viewwise_gated_ft_transformer_metadata.json`
- `models/metrics/viewwise_gated_ft_gate_summary.csv`

## 运行方式

```bash
python scripts/compare_viewwise_gated_transformers.py
```

## 解释建议

- 如果它优于标准 FT，就说明“按视图聚合后再门控融合”比简单 token 级结构编码更有效。
- 如果它仍弱于 `Multi-View Gated MLP`，则说明 Transformer backbone 还不是当前最合适的主干。

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

- 当前结论：
  - `View-Wise Gated FT-Transformer` 已经略微超过调参后的标准 FT 的 `macro-F1`
  - 它仍然落后于 `Multi-View Gated MLP`
  - 这说明“view-wise pooling + gated fusion”这条方向是有效的，比单纯 token 级结构编码更合理
  - 下一步如果继续推这条线，重点应放在：
    - 更好的 view pooling 方式
    - gate 稳定性
    - 与 multiview MLP 的 hybrid 融合
