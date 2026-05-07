# run_balanced_rf_baseline.py

## 作用

对应脚本：[scripts/bagging/run_balanced_rf_baseline.py](/media/990Pro/ML/scripts/bagging/run_balanced_rf_baseline.py)

这个脚本负责把 `BalancedRandomForestClassifier` 作为不平衡类别友好的 baseline 接入当前项目。

## 为什么值得做

- 当前数据中 `Enrolled` 是明显少数类
- 普通 RF 已经很强
- Balanced RF 是低成本、很自然的下一步

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 输出

- `models/metrics/balanced_rf_metrics.csv`
- `models/metrics/balanced_rf_metadata.json`
- `reports/figures/cm_balanced_random_forest.png`

## 运行方式

```bash
python scripts/bagging/run_balanced_rf_baseline.py
```

## 解释建议

- 如果它提高 `macro-F1`，说明少数类处理确实是关键。
- 如果它不如普通 RF，也依然有价值，因为它帮助确认“类不平衡并不是当前主瓶颈”。
