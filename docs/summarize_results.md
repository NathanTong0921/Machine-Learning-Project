# summarize_results.py

## 作用

对应脚本：[scripts/reporting/summarize_results.py](/media/990Pro/ML/scripts/reporting/summarize_results.py)

这个脚本负责把 baseline 和高级模型的结果统一整理成报告可直接使用的资产：

- 一张统一的模型对比表
- 一张 2x2 的混淆矩阵拼图
- 一份汇总 CSV

它不重新训练模型，只消费已有结果文件。

## 输入

- `models/metrics/classification_metrics.csv`
- `models/metrics/advanced_metrics.csv`
- `reports/figures/cm_decision_tree.png`
- `reports/figures/cm_random_forest.png`
- `reports/figures/cm_svm_rbf.png`
- `reports/figures/cm_multiview_gated_mlp.png`

## 输出

- `models/metrics/model_comparison_summary.csv`
- `reports/figures/model_comparison_table.png`
- `reports/figures/confusion_matrix_grid.png`

## 运行方式

```bash
python scripts/reporting/summarize_results.py
```

## 当前用途

- 把 4 个模型统一放进一张表，避免手工拼接
- 生成 2x2 混淆矩阵总图，便于报告展示
- 作为“实验结果 -> 报告材料”的稳定中间层

## 记录建议

- 如果以后加入第五个模型，不要直接覆盖当前 2x2 版式，应先决定是否改成新的排版。
- 如果修改指标列，也要同步更新本文件和表格标题。
