# compare_tabm_style_models.py

## 作用

对应脚本：[scripts/mlp/compare_tabm_style_models.py](/media/990Pro/ML/scripts/mlp/compare_tabm_style_models.py)

这个脚本负责比较：

- 当前最优 `Multi-View Gated MLP`
- 新实现的 `TabM-style Multi-Expert MLP`

## 当前实现思路

这个 `TabM-style` 版本不是官方 `tabm` 包的逐字复刻，而是一个贴合当前项目结构的本地实现：

1. 保留 `Multi-View Gated MLP` 的多视图编码与门控融合主干  
2. 在共享主干后引入多个 expert prediction heads  
3. 通过共享主干参数 + expert-specific modulation 来模拟参数高效 ensemble  
4. 训练时直接优化 expert 集合的平均预测

## 为什么这样做

- 它和当前项目最强主线最接近
- 改动成本比完全接入外部 `tabm` 包更低
- 更容易判断“多 expert”本身是否给这份数据带来增益

## 输出

- `models/metrics/tabm_style_comparison.csv`
- `reports/figures/tabm_style_comparison.png`
- `reports/figures/tabm_style_confusion_grid.png`
- `reports/figures/cm_tabm_style_multiexpert_mlp.png`
- `models/saved_models/tabm_style_multiexpert_mlp.pt`
- `models/metrics/tabm_style_multiexpert_mlp_metadata.json`
- `models/metrics/tabm_style_multiexpert_gate_summary.csv`

## 运行方式

```bash
python scripts/mlp/compare_tabm_style_models.py
```

## 解释建议

- 如果它优于当前最优 `Multi-View Gated MLP`，说明多 expert 的 ensemble-like 方向是正收益。
- 如果它没有超过，也不代表 TabM 路线失败，只说明当前这个 project-local 版本还不够成熟。

## Latest Recorded Run

- 日期：2026-04-22
- 对比结果：
  - `multiview_gated_mlp_best`
    - `accuracy = 0.7575`
    - `macro-F1 = 0.7217`
    - `AUC = 0.8925`
  - `tabm_style_multiexpert_mlp`
    - `accuracy = 0.7636`
    - `macro-F1 = 0.7186`
    - `AUC = 0.8888`

- 当前结论：
  - `TabM-style multi-expert MLP` 在 `accuracy` 上已经高于当前最优 `Multi-View Gated MLP`
  - 但 `macro-F1` 和 `AUC` 仍略低
  - 这说明“多 expert”方向是有效的，但当前 project-local 版本还没有完全赢过主线模型
