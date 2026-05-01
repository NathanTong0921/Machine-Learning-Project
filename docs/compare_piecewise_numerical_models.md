# compare_piecewise_numerical_models.py

## 作用

对应脚本：[scripts/compare_piecewise_numerical_models.py](/media/990Pro/ML/scripts/compare_piecewise_numerical_models.py)

这个脚本负责比较：

- 当前最优 `Multi-View Gated MLP`
- 调参后的 `FT-Transformer`
- 新实现的 `Piecewise FT-Transformer`
- 已跑通的 `TabPFN`（若结果文件存在）

## 当前实现思路

这版 `Piecewise FT-Transformer` 的重点不在 backbone，而在数值特征 embedding：

1. 对每个数值特征按训练集分位数建立 piecewise bins  
2. 把标量值映射成 piecewise linear basis  
3. 再通过可学习投影变成 token embedding  
4. 其余 transformer 主干保持与 tuned FT 基本一致

## 为什么这样做

- 这条线直接对应数值 embedding 论文中的核心启发
- 比继续放大标准 FT 更低成本，也更容易解释
- 能直接回答“数值 embedding 是否比 backbone 改动更重要”

## 输出

- `models/metrics/piecewise_numerical_comparison.csv`
- `reports/figures/piecewise_numerical_comparison.png`
- `reports/figures/piecewise_numerical_confusion_grid.png`
- `reports/figures/cm_piecewise_ft_transformer.png`
- `models/saved_models/piecewise_ft_transformer.pt`
- `models/metrics/piecewise_ft_transformer_metadata.json`

## 运行方式

```bash
python scripts/compare_piecewise_numerical_models.py
```

## 解释建议

- 如果它优于 tuned FT，说明数值 embedding 改动是有效的。
- 如果它接近或超过 `Multi-View Gated MLP`，说明这条低成本增强路线值得重点继续。
- 如果提升有限，则说明当前瓶颈不只在数值 embedding。

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
  - `piecewise_ft_transformer`
    - `accuracy = 0.6965`
    - `macro-F1 = 0.6655`
    - `AUC = 0.8666`
  - `tabpfn`
    - `accuracy = 0.7861`
    - `macro-F1 = 0.7131`
    - `AUC = 0.9017`

- 当前结论：
  - 这版 `piecewise numerical embedding` 没有带来提升
  - 相比 tuned FT，它在 `accuracy / macro-F1 / AUC` 上都明显更差
  - 因此当前最合理的判断是：在这份数据上，当前瓶颈并不主要来自“简单数值 tokenizer”
