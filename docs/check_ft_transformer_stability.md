# check_ft_transformer_stability.py

## 作用

对应脚本：[scripts/transformer/check_ft_transformer_stability.py](/media/990Pro/ML/scripts/transformer/check_ft_transformer_stability.py)

这个脚本负责对当前最佳 `FT-Transformer` 配置做 seed 稳定性测试。

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/ft_seed_stability.csv`
- `models/metrics/ft_seed_stability.md`
- `models/metrics/ft_seed_stability_summary.json`
- `reports/figures/ft_seed_stability_table.png`

## 当前测试协议

- 配置固定为当前最佳 FT：
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
python scripts/transformer/check_ft_transformer_stability.py
```

## 解释建议

- 重点看 `macro-F1` 的均值和标准差。
- 如果标准差明显偏大，就说明 FT 对初始化或数据划分很敏感。
- 如果均值明显低于 `Multi-View Gated MLP`，即使单次最好结果接近，也不能说 FT 更强。
