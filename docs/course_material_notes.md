# Course Material Notes

## 作用

这个文件用于把课程讲义内容映射到当前项目实现，回答两个问题：

1. 课程在这个项目里最希望你们体现哪些基础模型和方法  
2. 当前代码已经覆盖了哪些，哪些还可以继续补

## 课程材料里的核心主线

根据 `reference/course_material/` 中的讲义，和当前项目最相关的内容主要有：

### 监督学习

- `L02-L04`
- 重点：
  - 线性回归
  - 线性分类
  - 多分类
  - logistic regression
  - SVM
  - 多项式特征

### 特征工程与模型选择

- `L05-L06`
- 重点：
  - train / validation / test
  - overfitting / underfitting
  - bias-variance trade-off
  - precision / recall / F1 / ROC / AUC
  - regularization
  - feature selection
  - cross-validation

### 集成学习

- `L07`
- 重点：
  - decision tree
  - bagging
  - random forest
  - boosting
  - gradient boosting

### 聚类

- `L08`
- 重点：
  - k-means
  - hierarchical / agglomerative clustering
  - external metrics
  - internal metrics

## 和当前项目代码的对应关系

### 已经实现

- 监督学习 baseline：
  - `LogisticRegression`
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
  - `AdaBoostClassifier`
  - `ExtraTreesClassifier`
  - `SVC`
- 特征工程 baseline：
  - `LogisticRegression + PolynomialFeatures`
  - `LogisticRegression + RBF feature map (Nystroem)`
- 聚类：
  - `KMeans`
  - `AgglomerativeClustering`
- 模型评估：
  - accuracy
  - precision / recall / F1
  - ROC / AUC
  - confusion matrix
- 高级模型：
  - `Multi-View Gated MLP`
  - `FT-Transformer`
  - `View-Wise Gated FT`

### 课程里提到但还没有系统补齐

- 更正式的 cross-validation 入口
- 更明确的 feature selection 流程
- 更系统的 validation 报告

## 对当前项目的直接启发

### 1. 课程不会只接受“高级模型更强”这种叙事

你需要用课程里讲过的基础模型证明：

- 简单线性模型能做到什么程度
- 树模型为什么更强
- 高级模型到底在哪些指标上更有价值

所以 `logistic regression + SVM + decision tree + random forest` 这条线很重要。

### 2. 课程非常强调模型选择而不是只报单次成绩

`L05-L06` 很强调：

- train / validation / test 分离
- 过拟合判断
- error metric 的选择
- cross-validation

这意味着你报告里应该明确写：

- 为什么 `macro-F1` 比单纯 accuracy 更重要
- 为什么要做 seed 检查和多 trial
- 为什么某些模型虽然 accuracy 高，但 macro-F1 不一定最好

### 3. 集成学习是课程主线中的强基线

`L07` 明确说明：

- 单棵树不稳定
- random forest 是很强的稳健默认方法
- boosting 往往性能更高，但更复杂

这对你当前项目的实际含义是：

- `Random Forest` 应该被视为课程语境下的核心强 baseline
- 高级模型即使不全面超过 RF，也至少要在某些指标上给出明确价值

### 3.5. 除了当前 4 个基础模型，还可以自然实现什么

结合课程讲义，当前最自然还能补的模型有：

- `LogisticRegression + PolynomialFeatures`
- `LogisticRegression + RBF feature map`
- `GradientBoostingClassifier`
- `AdaBoostClassifier`
- `ExtraTreesClassifier`

其中前两者最适合作为“课程式特征工程 baseline”，后面三者更适合作为“课程式 ensemble 扩展 baseline”。

### 4. 聚类部分更重“评估解释”而不是只跑算法

`L08` 里非常强调：

- external metric
- internal metric
- clustering quality interpretation

所以项目里聚类不应只给结果图，还应明确说：

- 为什么这个聚类更合理
- 它和真实标签之间关系如何
- 是否存在过聚类 / 欠聚类

## 推荐的课程对齐实现顺序

1. `scripts/run_baselines.py`
   - 对齐 `L02-L07`
2. `scripts/run_clustering.py`
   - 对齐 `L08`
3. `scripts/compare_advanced_models.py`
   - 把高级模型放回课程 baseline 语境中看
4. 更进一步的 seed / tuning / structured-transformer 试验
   - 作为开放探索部分

## 当前判断

- 从课程要求看，你现在最强的主线叙事仍然应该是：
  - 基础模型：logistic / SVM / tree / RF / boosting / extra trees
  - 课程式增强 baseline：poly / RBF feature engineering
  - 开放探索：Multi-View Gated MLP 及 Transformer 变体
- 从当前结果看：
  - `CatBoost (10-fold tuned)` 目前是总体上最强、最稳的传统表格模型
  - `Multi-View Gated MLP` 仍然是最强的项目内高级模型主线之一
  - `TabPFN` 在 `accuracy / AUC` 上很强
  - Transformer 线有研究价值，但还没成为主线赢家
