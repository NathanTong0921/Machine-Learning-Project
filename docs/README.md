# Docs

这个目录用于存放与项目相关的说明材料，而不是代码或中间结果。

## 记录规则

- 每新增一个主流程脚本，都要在 `docs/` 下新增一份对应说明文档。
- 说明文档的目标不是重复代码，而是记录：
  - 这个脚本为什么存在
  - 输入和输出是什么
  - 运行顺序是什么
  - 会产出哪些文件
  - 结果应该如何解释
- 推荐命名方式：
  - 脚本：`scripts/run_xxx.py`
  - 文档：`docs/run_xxx.md`

## 当前文档

- `DSAA2011-26sp-project.pdf`：课程项目原始要求
- `IDEA_REPORT.md`：模型方向筛选与推荐路线
- `data_dictionary.md`：数据字段分组与清洗注意事项
- `course_material_notes.md`：课程讲义与当前项目实现的映射说明
- `DECLARATION_AND_TIPS.md`：项目声明、使用建议和脚本顺序
- `prepare_data.md`：数据清洗与拆分脚本说明
- `run_clustering.md`：聚类脚本说明
- `run_baselines.md`：baseline 分类脚本说明
- `run_feature_engineered_baselines.md`：特征工程 baseline 脚本说明
- `run_tabpfn_baseline.md`：TabPFN 外部 baseline 脚本说明
- `run_catboost_baseline.md`：CatBoost baseline 脚本说明
- `run_lightgbm_baseline.md`：LightGBM baseline 脚本说明
- `tune_lightgbm_baseline.md`：LightGBM 10-fold 超参数搜索说明
- `run_xgboost_baseline.md`：XGBoost baseline 脚本说明
- `run_balanced_rf_baseline.md`：BalancedRandomForest baseline 脚本说明
- `tune_catboost_baseline.md`：CatBoost 10-fold 超参数搜索说明
- `tabpfn_blocker.md`：TabPFN 官方 license/token 阻塞说明
- `search_and_ensemble_baselines.md`：baseline 暴力搜参与集成脚本说明
- `summarize_primary_contenders.md`：当前主力模型总表汇总说明
- `run_advanced.md`：高级模型脚本说明
- `summarize_results.md`：结果汇总脚本说明
- `compare_advanced_models.md`：高级模型正式对比脚本说明
- `tune_ft_transformer.md`：FT-Transformer 小范围调参脚本说明
- `check_ft_transformer_stability.md`：FT-Transformer seed 稳定性测试说明
- `compare_structured_tabular_transformers.md`：结构感知 Transformer 正式对比说明
- `compare_viewwise_gated_transformers.md`：view-wise pooling 与 gated fusion Transformer 对比说明
- `check_viewwise_gated_ft_stability.md`：View-Wise Gated FT seed 稳定性测试说明
- `compare_attention_pooled_viewwise_transformers.md`：attention pooling 的 view-wise gated Transformer 对比说明
- `compare_tabm_style_models.md`：TabM-style multi-expert MLP 对比说明
- `compare_trompt_style_models.md`：Trompt-style prompt transformer 对比说明
- `compare_piecewise_numerical_models.md`：piecewise numerical embedding 对比说明
- `theory_multiview_gated_mlp.md`：多视图门控模型的理论主线
- `future_proposal.md`：后续提案方向的优先级草案
- `student_dropout_model_survey.md`：学生流失三分类表格项目的模型调研与优先级建议

## 建议内容

- 课程项目要求 PDF
- 数据说明与字段解释
- 实验设计笔记
- 模型选择与阅读笔记
- 每个主流程脚本的运行说明
- 最终报告所引用的辅助材料
