# IDEA_REPORT

## 项目

- 方向：Student Dropout Dataset 的三分类预测与课程项目扩展
- 日期：2026-04-21
- 数据规模：4424 条样本，36 个输入特征，标签为 `Dropout / Enrolled / Graduate`

## 数据要点

- 这是中小规模表格分类问题，不适合盲目上大模型。
- 特征是“类别编码 + 连续值 + 学期成绩统计”的混合结构。
- 标签轻度不平衡：`Graduate 2209`，`Dropout 1421`，`Enrolled 794`。
- 互信息最高的特征主要来自两学期成绩与通过门数，其次才是缴费、年龄、课程等背景变量。
- 这说明“学业轨迹”比纯人口统计特征更关键，模型设计应显式利用这一点。

## Top Ideas

### 1. 多视图门控网络（推荐）

- 标题：Multi-View Gated MLP for Student Outcome Prediction
- 核心想法：把特征拆成 4 个视图分别编码，再由门控层自适应融合。
- 视图划分：
  - 入学/人口学背景
  - 家庭社会经济背景
  - 第一学期表现
  - 第二学期表现 + 宏观经济变量
- 为什么适合这份数据：
  - 数据天然分块，直接把所有列平铺进单个模型会丢掉结构信息。
  - 可以可视化每个样本上哪一类视图贡献最大，报告更有内容。
  - 相比普通 MLP，更容易讲清楚“学生结果是如何被不同信息源共同决定的”。
- 预期优势：
  - 比 logistic regression/普通 MLP 更能建模跨视图交互。
  - 比纯 boosting 更有“方法设计”味道，适合开放探索部分。
- 风险：
  - 小数据集上网络容易过拟合，需要 dropout、weight decay、early stopping。

### 2. FT-Transformer / TabTransformer

- 标题：Transformer for Mixed-Type Tabular Prediction
- 核心想法：把低基数类别特征做 embedding，数值特征线性投影，再用 self-attention 建模特征间关系。
- 为什么适合：
  - 这份数据有不少低到中等基数类别列，如 `Course`、`Application mode`、父母职业/学历。
  - attention 权重可以帮助展示“哪些特征对某类学生更重要”。
- 预期优势：
  - 比传统线性模型和树模型更擅长学习高阶特征交互。
  - 很适合作为“有意思但仍合理”的开放探索模型。
- 风险：
  - 数据只有 4424 条，Transformer 不一定稳定优于强树模型。
  - 需要更谨慎的超参和更强正则。

### 3. TabNet

- 标题：Sequential Attentive Feature Selection for Student Risk Modeling
- 核心想法：通过逐步选择重要特征完成预测，自带一定解释性。
- 为什么适合：
  - 课程项目通常需要“模型效果 + 解释”，TabNet 两者兼顾。
  - 可以展示不同 decision step 关注了哪些字段。
- 预期优势：
  - 训练后容易做 feature mask 可视化。
  - 作为非树型、非线性 tabular 模型，叙事完整。
- 风险：
  - 实际表现波动比较大，常常需要较多调参。

## 不建议作为主线的方法

- 纯 RNN/LSTM：数据不是时间序列，只是两学期汇总统计，序列长度太短。
- 图神经网络：除非你们先构造合理的学生相似图，否则更像“硬套模型”。
- 普通 MLP：能做 baseline，但方法亮点不够。
- 只做 CatBoost/XGBoost/LightGBM：效果可能强，但你已经明确不想停留在标准方案。

## 推荐主线

### 推荐方案：Idea 1

- 结论：优先做 **多视图门控网络**，再配一个 **FT-Transformer 或 TabNet** 作为开放探索增强。
- 原因：
  - 最贴合这份数据的结构性。
  - 方法新颖度比标准 boosting 更强，但实现仍在课程项目可控范围内。
  - 容易产出好看的图：视图权重、特征重要性、混淆矩阵、ROC、t-SNE。

## 最小实验矩阵

- Baseline 1：Decision Tree
- Baseline 2：Random Forest 或 SVM
- Main Model：Multi-View Gated MLP
- Exploration Model：FT-Transformer 或 TabNet

这样既满足课程“至少两个模型”的要求，也能在开放探索部分体现进阶设计。

## 具体实现建议

### 预处理

- 清理列名中的 BOM、`\t`、`\r`
- 把低基数列标为 categorical
- 数值列标准化
- 使用 stratified train/test split
- 由于 `Enrolled` 最少，训练时使用 class weight 或 focal loss

### 评估

- 主要看 macro-F1、weighted-F1、per-class recall
- 不能只看 accuracy，因为 `Enrolled` 类较少
- ROC/AUC 建议做 one-vs-rest 多分类版本
- 混淆矩阵重点看：
  - `Dropout` 是否常被错判成 `Enrolled`
  - `Enrolled` 是否常被错判成 `Graduate`

### 开放探索亮点

- 做视图消融：只用背景、只用第一学期、只用第二学期、全部融合
- 做解释分析：
  - Multi-view gate 权重分布
  - TabNet/Transformer 的 feature importance 或 attention
- 做 class imbalance 对比：
  - 无 class weight
  - 有 class weight
  - focal loss

## 结论

- 最值得做的不是“更强的标准 boosting”，而是 **能显式利用学生多视图特征结构** 的模型。
- 如果你想要“有意思、能讲故事、实现不离谱”的方案，首选：
  - `Multi-View Gated MLP`
  - `FT-Transformer`
  - `TabNet`

- 推荐最终路线：
  - 必做任务部分：Decision Tree + Random Forest/SVM
  - 开放探索部分：Multi-View Gated MLP
  - 如果还有余力：再加 FT-Transformer 做对比
