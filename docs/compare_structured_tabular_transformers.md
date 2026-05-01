# compare_structured_tabular_transformers.py

## 作用

对应脚本：[scripts/compare_structured_tabular_transformers.py](/media/990Pro/ML/scripts/compare_structured_tabular_transformers.py)

这个脚本负责比较三条路线：

- 当前最优 `Multi-View Gated MLP`
- 调过一轮的标准 `FT-Transformer`
- 新的 `View-Aware FT-Transformer`

## 为什么存在

它回答两个更有研究价值的问题：

1. 改 FT 的输入表示是否有效  
2. 比起继续放大标准 FT，结构感知变体是否更适合这份数据

## 新变体包含什么

### 输入表示改动

- 数值特征不再只用简单线性 tokenizer
- 改成带 periodic embedding 的数值 token 化

### 结构感知改动

- 给 token 加入 `column embedding`
- 给 token 加入 `view embedding`
- 给 token 加入 `type embedding`

其中 `view` 对应当前数据的天然分组：

- 入学背景
- 家庭财务
- 第一学期
- 第二学期与宏观变量

## 输出

- `models/metrics/structured_transformer_comparison.csv`
- `reports/figures/structured_transformer_comparison.png`
- `reports/figures/structured_transformer_confusion_grid.png`
- `reports/figures/cm_view_aware_ft_transformer.png`
- `models/saved_models/view_aware_ft_transformer.pt`
- `models/metrics/view_aware_ft_transformer_metadata.json`

## 运行方式

```bash
python scripts/compare_structured_tabular_transformers.py
```

## 解释建议

- 如果 `View-Aware FT-Transformer` 优于标准 FT，就说明“结构对齐”比“单纯放大标准 FT”更重要。
- 如果它仍打不过 `Multi-View Gated MLP`，也不代表这条线失败，而是说明结构化 Transformer 还需要继续设计。

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
  - `view_aware_ft_transformer`
    - `accuracy = 0.7229`
    - `macro-F1 = 0.6870`
    - `AUC = 0.8814`

- 当前结论：
  - 这版 `View-Aware FT-Transformer` 没有超过调参后的标准 FT
  - 说明“改 token 化 + 加 view/type/column embedding”这条方向本身合理，但当前实现还不够成熟
  - 从结果看，结构信息的显式编码没有自动转化成更强性能，可能需要更合适的融合方式，而不是简单把所有结构信号直接加到 token 上
