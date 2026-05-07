# run_advanced.py

## 作用

对应脚本：[scripts/mlp/run_advanced.py](/media/990Pro/ML/scripts/mlp/run_advanced.py)

这个脚本预留给非标准表格模型，例如：

- `Multi-View Gated MLP`
- `FT-Transformer`
- `TabNet`

它用于承接课程项目中的开放探索部分，而不是替代 baseline。

当前已经实现的主线模型是 **Multi-View Gated MLP**。

## 输入

- `data/processed/train.csv`
- `data/processed/test.csv`

## 预期输出

- `models/metrics/advanced_metrics.csv`
- `models/metrics/multiview_gate_summary.csv`
- `models/metrics/advanced_training_history.csv`
- `models/metrics/advanced_metadata.json`
- `reports/figures/` 下的高级模型相关图
- `models/saved_models/` 下的模型权重或检查点

当前具体输出：

- `models/metrics/advanced_metrics.csv`
- `models/metrics/multiview_gate_summary.csv`
- `models/metrics/advanced_training_history.csv`
- `models/metrics/advanced_metadata.json`
- `reports/figures/cm_multiview_gated_mlp.png`
- `models/saved_models/multiview_gated_mlp.pt`

## 当前状态

- 当前实现模型：`Multi-View Gated MLP`
- 视图划分：
  - `enrollment_profile`
  - `family_finance`
  - `first_semester`
  - `second_semester_macro`
- 训练方式：
  - 使用 `train.csv` 再切出验证集
  - 使用 class weight 缓解类不平衡
  - 使用 early stopping 选择最佳 epoch

## Latest Recorded Run

- 日期：2026-04-21
- 主模型：`Multi-View Gated MLP`
- 测试集指标：
  - `accuracy = 0.7508`
  - `macro-F1 = 0.7180`
  - `weighted-F1 = 0.7627`
  - `macro ROC-AUC (OvR) = 0.8910`

- 当前观察：
  - `Graduate` 和 `Dropout` 识别较稳
  - `Enrolled` 仍然最难，但召回率已经达到 `0.70`
  - gate summary 显示 `second_semester_macro` 在三类样本上都占主导，说明第二学期学业表现是最强信号

- 对应结果文件：
  - `models/metrics/advanced_metrics.csv`
  - `models/metrics/multiview_gate_summary.csv`
  - `models/metrics/advanced_training_history.csv`
  - `models/metrics/advanced_metadata.json`
  - `reports/figures/cm_multiview_gated_mlp.png`
  - `models/saved_models/multiview_gated_mlp.pt`

## 运行方式

```bash
python scripts/mlp/run_advanced.py
```

## 记录建议

- 每实现一个高级模型，就把模型结构、输入视图、训练策略、输出文件写进本文件。
- 如果最终只选择一个高级模型作为主线，也要在这里写明放弃其他方案的原因。
- 如果修改视图划分、hidden dimension、dropout、学习率，也必须同步更新这里。
