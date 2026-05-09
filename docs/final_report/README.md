# Final Report Materials

这个目录用于整理最终汇报材料，重点服务 PPT 和报告写作。

## 文件结构

- `analysis_themes.md`: 五个重点分析主题的结论草案，包括 GBDT、ensemble、MLP 编码、Course 171 异常子群、SVM 去噪。
- `ppt_figure_proposals.md`: 建议制作的 PPT 图片、每张图要表达的信息、数据来源和优先级。
- `boost_denoise_check.md`: 验证 SVM 去噪策略迁移到 boosting 模型时是否仍然有效。
- `moe_v2_cross_seed_check.md`: supervised error-router MoE v2 的跨 seed 稳定性和 Core3 member 对比。

## 汇报主线建议

本项目不建议只讲“哪个模型分数最高”，更适合包装成：

> 从全局建模性能，到局部机制发现，再到不确定样本的决策策略设计。

也就是：

1. 用 GBDT / ensemble 建立强基线。
2. 用编码实验解释“特征表示会改变最优解”。
3. 用 Course 171 子群证明模型还能发现制度机制，而不是只输出黑盒标签。
4. 用 SVM 去噪展示 data-centric AI 的可解释提分。
5. 最后强调：本项目的价值不只是 Macro-F1，而是可复现、可解释、可决策。
