# Declaration And Tips

## Declaration

- 本项目以课程要求为边界，核心机器学习实现、实验设计、结果分析应由项目成员自己完成。
- GenAI 可以用于辅助整理思路、修正文案、生成代码骨架，但不应替代你们对方法与结果的理解。
- 若最终报告或展示材料使用了 AI 辅助，应按课程要求明确披露使用范围。

## Practical Tips

- 不要只看 `accuracy`，这份数据是三分类且 `Enrolled` 更少，`macro-F1` 和每类 `recall` 更重要。
- 不要把所有整数列都当连续变量，很多字段实际上是类别编码。
- 聚类结果要同时看内部指标和与真实标签的对应关系，单一指标不够。
- 如果高级模型没有明显优于 baseline，不要硬讲“更先进就更好”，要解释数据规模和归纳偏置的原因。
- 报告里最好加入失败尝试，这会让分析更可信。

## Suggested Script Order

1. `python scripts/preprocessing/prepare_data.py`
2. `python scripts/clustering/run_clustering.py`
3. `python scripts/baseline/run_baselines.py`
4. `python scripts/mlp/run_advanced.py`
