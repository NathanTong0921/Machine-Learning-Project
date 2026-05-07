# summarize_primary_contenders.py

## 作用

对应脚本：[scripts/reporting/summarize_primary_contenders.py](/media/990Pro/ML/scripts/reporting/summarize_primary_contenders.py)

这个脚本把当前最重要的几条主线汇总到一张总表中：

- `RandomForest`
- `voting_soft`
- `Multi-View Gated MLP`
- `TabPFN`
- `CatBoost`

## 输出

- `models/metrics/primary_contenders_summary.csv`
- `reports/figures/primary_contenders_summary.png`

## 运行方式

```bash
python scripts/reporting/summarize_primary_contenders.py
```
