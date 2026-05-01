# run_tabpfn_baseline.py

## 作用

对应脚本：[scripts/run_tabpfn_baseline.py](/media/990Pro/ML/scripts/run_tabpfn_baseline.py)

这个脚本负责把 `TabPFN` 作为一个外部 fancy baseline 接入当前项目。

## 为什么值得做

- `TabPFN` 是当前最有代表性的 tabular foundation model 路线之一
- 对中小规模表格分类问题尤其值得快速验证
- 它能帮助判断：当前项目是否值得继续往 foundation-model 方向扩展

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/tabpfn_metrics.csv`
- `models/metrics/tabpfn_metadata.json`
- `reports/figures/cm_tabpfn.png`

## 运行方式

```bash
python scripts/run_tabpfn_baseline.py
```

## 注意事项

- TabPFN 首次使用时可能需要：
  - 接受官方 license
  - 下载模型权重
  - 配置 token 或浏览器认证

- 因此它和项目内自实现模型不同，可能出现“环境/认证阻塞”而不是算法报错。

## 解释建议

- 如果 TabPFN 明显优于当前 classical baselines，说明 foundation-model 路线值得认真考虑。
- 如果它不如 `Multi-View Gated MLP`，也仍然有价值，因为它是一个很强的外部参考线。
