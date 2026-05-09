# PPT Figure Proposals

以下图片建议按“主线图优先、结果图次之、机制图作为亮点”组织。每张图都标注了推荐用途和数据来源。

## Priority A: 必做图

### Figure 1. 项目总主线图

- 图型：横向流程图。
- 内容：`Data preprocessing -> Feature representation -> Model families -> OOF ensemble -> Mechanism discovery -> Decision strategy`。
- 目的：开场说明项目不是单纯调参，而是完整建模流程。
- 建议标题：`From Prediction to Mechanism-Aware Student Risk Modeling`。

### Figure 2. GBDT 为什么赢

- 图型：柱状图 + 小型 boosting 示意图。
- 数据来源：`models/metrics/boost/classic_boosting_cross_seed_metrics.csv`。
- 展示指标：accuracy、macro-F1、AUC、Enrolled F1。
- 高亮：`gbdt_log_loss`, seed 62, macro-F1 0.7501, AUC 0.9058。
- 目的：说明 GBDT 在非线性表格数据上能通过残差迭代修正困难区域。
- 注意：图注标明这是 cross-seed 中的高分 split，不和固定 split 结果混为同一口径。

### Figure 3. 最佳 ensemble 的 OOF 流程

- 图型：流程图。
- 数据来源：`models/metrics/ensemble/oof_probability_ensemble_semester_members_metrics.csv`。
- 内容：fold 训练 -> OOF probability -> weight search -> held-out test。
- 目的：强调无泄漏评估和概率融合。
- 建议配一个结果小表：
  - accuracy 0.7824
  - macro-F1 0.7343
  - AUC 0.9013
  - Enrolled F1 0.5487

### Figure 4. MLP 编码方式对比

- 图型：分组柱状图。
- 数据来源：`models/metrics/mlp/multiview_application_order_encoding_tuned_summary.csv`。
- X 轴：scalar tuned、thermometer tuned、target encoding OOF tuned。
- Y 轴：macro-F1、AUC、Enrolled acc。
- 目的：展示编码不是细节，而是改变模型假设。
- 建议旁边放一个小示意：
  - scalar: `order = 5`
  - thermometer: `[1,1,1,1,1,0,0,0,0]`
  - target encoding: OOF class-rate/probability representation

### Figure 5. Course 171 子群组成图

- 图型：堆叠条形图或 donut chart。
- 数据来源：`docs/findings/course_171_ctesp_creditation_subgroup.md`。
- 数据：Graduate 75, Dropout 77, Enrolled 28。
- 目的：一眼证明这个 outlier 子群不是纯 Graduate，因此不能硬覆盖。
- 建议标题：`Course 171 + zero curricular activity is a mechanism subgroup, not a clean label rule`。

### Figure 6. SVM 去噪前后对比

- 图型：before/after 柱状图。
- 数据来源：`docs/feature/svm_noise_feature_engineering.md`。
- 展示指标：accuracy、macro-F1、Enrolled recall、Enrolled F1。
- 重点 delta：
  - macro-F1 +0.0057
  - Enrolled recall +0.0126
  - Enrolled F1 +0.0128
- 目的：展示 data-centric AI 的小而稳提分。

## Priority B: 很适合展示的亮点图

### Figure 7. 三类学生的难度结构

- 图型：class-wise precision/recall/F1 雷达图或热力表。
- 数据来源：GBDT、OOF ensemble、SVM denoise、MLP thermometer 的详细 metrics。
- 目的：说明 Enrolled 是主要困难类，不能只看总 accuracy。
- 推荐信息：Dropout 和 Graduate 通常更稳，Enrolled recall/F1 决定 macro-F1 上限。

### Figure 8. Course 171 机制发现流程

- 图型：四步流程图。
- 内容：
  1. 模型/可视化发现异常局部区域。
  2. 定义 `Course == 171` + zero curricular subgroup。
  3. 查制度背景，提出 CTeSP / credit recognition 解释。
  4. 验证不能硬覆盖，只做 soft/report-only 策略。
- 目的：这是 open-ended highlight，非常适合最后展示“我们不只做模型，还解释数据机制”。

### Figure 9. Stacking 为什么失败

- 图型：OOF vs test 对比柱状图。
- 数据来源：`models/metrics/ensemble/cross_seed_stacking_metrics.csv`。
- 内容：OOF soft-average/stacking 看起来不错，但 test macro-F1 下降。
- 目的：展示实验纪律：不是所有复杂 ensemble 都有效，held-out test 才是最后标准。

### Figure 10. 表示学习和超参数交互

- 图型：两组柱状图。
- 数据来源：
  - `models/metrics/mlp/multiview_application_order_encoding_metrics.csv`
  - `models/metrics/mlp/multiview_application_order_encoding_tuned_summary.csv`
- 内容：未调参替换编码会下降；重调参后 thermometer 提升。
- 目的：强调“representation change requires re-tuning”。

## Priority C: 可选补充图

### Figure 11. 模型方法地图

- 图型：二维象限图。
- X 轴：interpretability / mechanism insight。
- Y 轴：predictive performance。
- 点：GBDT、OOF ensemble、MLP thermometer、SVM denoise、Course 171 rule、MoE。
- 目的：总结每个实验的定位，而不是让观众只盯最高分。

### Figure 12. 决策策略矩阵

- 图型：表格或 heatmap。
- 行：OOF ensemble、GBDT、MLP thermometer、Course 171 flag、SVM denoise。
- 列：performance gain、robustness、interpretability、risk of overfitting、deployment value。
- 目的：作为结尾页，说明最终方案如何选择。

### Figure 13. RBF SVM 去噪机制示意

- 图型：二维概念图。
- 内容：加入噪声维度后类别距离被稀释；移除低价值维度后边界更清晰。
- 目的：解释为什么 SVM 比树模型更依赖特征清洁度。

## 推荐 PPT 顺序

1. Project goal and evaluation protocol。
2. Overall modeling pipeline。
3. Baseline and model family comparison。
4. Why GBDT works best on this data。
5. Best OOF ensemble and why stacking was not adopted。
6. MLP encoding experiment: scalar vs thermometer vs target encoding。
7. Course 171 mechanism subgroup discovery。
8. SVM denoising and data-centric improvement。
9. Final decision matrix。
10. Takeaway: prediction + mechanism + decision design。
