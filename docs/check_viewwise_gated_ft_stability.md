# check_viewwise_gated_ft_stability.py

## 作用

对应脚本：[scripts/transformer/check_viewwise_gated_ft_stability.py](/media/990Pro/ML/scripts/transformer/check_viewwise_gated_ft_stability.py)

这个脚本负责对当前最佳 `View-Wise Gated FT-Transformer` 配置做 seed 稳定性测试。

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/viewwise_gated_ft_seed_stability.csv`
- `models/metrics/viewwise_gated_ft_seed_stability.md`
- `models/metrics/viewwise_gated_ft_seed_stability_summary.json`
- `reports/figures/viewwise_gated_ft_seed_stability_table.png`

## 当前测试协议

- 配置固定为当前最佳 `View-Wise Gated FT`：
  - `d_token = 48`
  - `n_blocks = 2`
  - `lr = 0.0008`
  - `attention_dropout = 0.20`
  - `ff_dropout = 0.20`
- seed 集合：
  - `42`
  - `62`
  - `82`

## 运行方式

```bash
python scripts/transformer/check_viewwise_gated_ft_stability.py
```

## 解释建议

- 重点看 `macro-F1` 的均值和标准差。
- 如果标准差不大且均值高于标准 FT 的 seed 均值，说明这条新变体改动不是偶然。
- 如果均值仍明显低于 `Multi-View Gated MLP`，则它更适合当次优研究线，而不是当前主线。

## Latest Recorded Run

- 日期：2026-04-21
- 每个 seed 结果：
  - `seed 42`: `accuracy = 0.7229`, `macro-F1 = 0.6927`, `AUC = 0.8796`
  - `seed 62`: `accuracy = 0.7417`, `macro-F1 = 0.7066`, `AUC = 0.8875`
  - `seed 82`: `accuracy = 0.7236`, `macro-F1 = 0.6907`, `AUC = 0.8778`

- 汇总：
  - `mean accuracy = 0.7294`
  - `std accuracy = 0.0087`
  - `mean macro-F1 = 0.6967`
  - `std macro-F1 = 0.0071`
  - `mean AUC = 0.8816`
  - `std AUC = 0.0042`

- 当前结论：
  - 波动不算大，说明这条新变体不是完全不稳定
  - 但从均值看，它还没有稳定优于标准 FT 的 seed 均值
  - 因此当前最合理的判断是：
    - `View-Wise Gated FT` 是一个有希望的次优研究方向
    - 但还不足以替代当前主线 `Multi-View Gated MLP`
