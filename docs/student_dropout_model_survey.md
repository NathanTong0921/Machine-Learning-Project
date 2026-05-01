# student_dropout_model_survey.md

## 作用

这份文档记录当前学生流失三分类表格项目的模型调研结论，重点覆盖：

- `Random Forest` 及其值得尝试的变种
- `XGBoost`、`LightGBM`、`CatBoost`
- 少量值得保留的非树型 classical 模型

本笔记只做研究记录，不对应某一个脚本，也不代表已经在项目内实现。

## 当前项目判断

基于本地项目与原始数据检查，这个任务有几个直接影响选型的特点：

- 数据规模为中小型表格任务：`4424` 条样本、`36` 个特征、三分类。
- 标签分布不均衡：`Graduate` 最多，`Enrolled` 最少，约为 `2209 / 1421 / 794`。
- 项目文档已明确很多整数列本质上是类别编码，不应默认视为连续变量。
- 当前 baseline 中，树模型已经很强：
  - `RandomForest` 测试集 `f1_macro = 0.7131`
  - `ExtraTrees` 测试集 `f1_macro = 0.7063`
  - 10-fold 搜索里 `ExtraTrees` 略高于 `RandomForest`

这些信号说明：下一轮最值得优先尝试的是更强的表格树模型，而不是继续增加普通深度模型复杂度。

## 与当前工程最相关的实现细节

- 当前预处理使用 `ColumnTransformer + StandardScaler + OneHotEncoder`：
  - 见 `src/preprocessing.py`
- 项目里已经显式维护了一份类别列清单：
  - 见 `src/config.py` 中的 `CATEGORICAL_COLUMNS`

这意味着如果后续尝试 `CatBoost` / `LightGBM` / `XGBoost` 的原生类别支持，最好不要再沿用“全部先进 OHE 再喂模型”的思路，而要显式利用类别列信息。

## 候选模型与结论

### 1. CatBoostClassifier

- 一手来源：
  - 官方文档：<https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier>
  - 类别特征文档：<https://catboost.ai/docs/en/features/categorical-features>
  - 原始论文：<https://papers.nips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html>
- 核心优势：
  - 原生处理类别特征
  - ordered boosting 用来降低 prediction shift / target leakage 风险
  - 小中型表格任务上通常非常稳
- 适合当前项目的原因：
  - 学生流失数据中有大量“编码后看起来像整数、实际是类别”的字段
  - 样本量不大，CatBoost 往往比手工 OHE 的 boosting 更稳定
  - 你当前最关心的是 `macro-F1` 和少数类表现，CatBoost 很适合作为首个强外部 baseline
- 潜在风险：
  - 如果类别列未正确传入 `cat_features`，优势会明显缩水
  - 使用类权重后，概率校准可能变差
  - 第二学期成绩类特征很强，模型可能学成“高精度但解释偏后验”
- 最小示例：

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1:average=Macro",
    auto_class_weights="Balanced",
    depth=6,
    learning_rate=0.03,
    iterations=500,
    verbose=0,
    random_seed=42,
)

model.fit(X_train, y_train, cat_features=cat_cols)
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `depth`
  - `learning_rate`
  - `iterations`
  - `l2_leaf_reg`
  - `random_strength`
  - `bagging_temperature`
  - `one_hot_max_size`
  - `auto_class_weights`

### 2. LGBMClassifier

- 一手来源：
  - 官方 sklearn API：<https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>
  - 官方高级主题：<https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html>
  - 原始论文：<https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf>
- 核心优势：
  - leaf-wise GBDT
  - GOSS 与 EFB 提升效率
  - 对中等规模表格通常又快又强
- 适合当前项目的原因：
  - 如果把整数类别列显式标成 categorical，LightGBM 往往很有竞争力
  - 相比 CatBoost，搜索速度通常更快，适合先大范围扫参数
  - 可以作为“强表格 boosting 基线”与 CatBoost/XGBoost 三方对照
- 潜在风险：
  - 小数据上更容易过拟合，`num_leaves` 太大时常伤 `macro-F1`
  - 官方文档明确提示 `class_weight` 可能让概率估计变差
  - 若类别列没处理好，常退化成“普通数值 boosting”
- 最小示例：

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective="multiclass",
    class_weight="balanced",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=400,
    random_state=42,
)

model.fit(X_train, y_train, categorical_feature=cat_cols)
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `num_leaves`
  - `max_depth`
  - `min_child_samples`
  - `learning_rate`
  - `n_estimators`
  - `subsample`
  - `colsample_bytree`
  - `reg_alpha`
  - `reg_lambda`
  - `cat_smooth`
  - `class_weight`

### 3. XGBClassifier

- 一手来源：
  - 官方 sklearn API：<https://xgboost.readthedocs.io/en/latest/python/python_api.html>
  - 官方参数文档：<https://xgboost.readthedocs.io/en/stable/parameter.html>
  - 官方 categorical tutorial：<https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html>
  - 原始论文：<https://arxiv.org/pdf/1603.02754>
- 核心优势：
  - 正则化强、工程成熟、鲁棒性高
  - `hist` 树方法与 early stopping 很好用
  - 常作为 tabular 任务必须有的强 baseline
- 适合当前项目的原因：
  - 三分类任务直接支持 `multi:softprob`
  - 可以稳定提供一个“很难太差”的 boosting 对照
  - 若你希望先用最熟悉的 boosting 框架做正式实验，它是最稳妥选项之一
- 潜在风险：
  - 原生 categorical 支持没有 CatBoost 那么“开箱即用”
  - 若类别列处理不规范，效果可能不如 CatBoost / LightGBM
  - 深树和高学习率容易伤少数类 recall
- 最小示例：

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="mlogloss",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=400,
    random_state=42,
)

model.fit(X_train, y_train)
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `max_depth`
  - `max_leaves`
  - `min_child_weight`
  - `gamma`
  - `subsample`
  - `colsample_bytree`
  - `reg_alpha`
  - `reg_lambda`
  - `learning_rate`
  - `n_estimators`
  - `tree_method`
  - `enable_categorical`

### 4. BalancedRandomForestClassifier

- 一手来源：
  - 官方文档：<https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html>
  - 原始技术报告：<https://statistics.berkeley.edu/tech-reports/666>
- 核心优势：
  - 继承 RF 的稳定性
  - 对不平衡类别做内建采样
  - 常对少数类 recall 和 `macro-F1` 更友好
- 适合当前项目的原因：
  - 你的 `Enrolled` 是明显少数类
  - 当前项目评价重点本就不应只看 accuracy
  - 如果普通 RF 已经强，Balanced RF 通常是低成本、很合理的下一步
- 潜在风险：
  - 不一定提高整体 accuracy
  - 多分类下 `sampling_strategy` 需要谨慎设定
  - 可能让决策边界更“激进”，导致多数类稳定性下降
- 最小示例：

```python
from imblearn.ensemble import BalancedRandomForestClassifier

model = BalancedRandomForestClassifier(
    n_estimators=400,
    sampling_strategy="all",
    replacement=True,
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_enc, y_train)
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `n_estimators`
  - `max_depth`
  - `min_samples_leaf`
  - `max_features`
  - `sampling_strategy`
  - `replacement`
  - `class_weight`

### 5. ExtraTreesClassifier

- 一手来源：
  - 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html>
  - 原始论文 DOI：<https://doi.org/10.1007/s10994-006-6226-1>
- 核心优势：
  - 更强随机化
  - 训练速度快
  - 对中小型表格常是极强 baseline
- 适合当前项目的原因：
  - 你项目里它已经是强基线之一
  - 10-fold 搜索结果里它略优于普通 RF
  - 后续所有新模型都应至少对齐它的表现
- 潜在风险：
  - 随机化更强，某些细粒度阈值结构上限可能不如 boosting
  - 如果只看 test 一次，容易误判“已经够好”
- 最小示例：

```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_enc, y_train)
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `n_estimators`
  - `max_depth`
  - `min_samples_leaf`
  - `max_features`
  - `bootstrap`
  - `class_weight`

### 6. RandomForestClassifier

- 一手来源：
  - 官方文档：<https://sklearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
  - 原始论文 DOI：<https://doi.org/10.1023/A:1010933404324>
- 核心优势：
  - 稳定、可解释、耐用
  - 对表格基线很可靠
- 适合当前项目的原因：
  - 当前测试集上已经很强
  - 可以作为后续所有复杂模型的稳定参照
- 潜在风险：
  - 上限通常低于强 boosting
  - 类别不平衡时只靠 `class_weight` 有时不够
- 最小示例：

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=10,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_enc, y_train)
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `n_estimators`
  - `max_depth`
  - `min_samples_leaf`
  - `max_features`
  - `class_weight`
  - `bootstrap`

### 7. HistGradientBoostingClassifier

- 一手来源：
  - 官方文档：<https://sklearn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>
- 核心优势：
  - sklearn 原生现代 boosting
  - 支持 early stopping
  - 支持 `class_weight`
- 适合当前项目的原因：
  - 如果你暂时不想引入新库，它是最值得补的 sklearn boosting baseline
  - 比旧的 `GradientBoostingClassifier` 更现代
- 潜在风险：
  - 在这个数据规模上未必一定赢过 ExtraTrees
  - 类别处理通常不如 CatBoost 路线自然
- 最小示例：

```python
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    loss="log_loss",
    class_weight="balanced",
    max_leaf_nodes=31,
    learning_rate=0.05,
    max_iter=400,
    early_stopping=True,
    random_state=42,
)

model.fit(X_train_proc, y_train)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `max_leaf_nodes`
  - `max_depth`
  - `min_samples_leaf`
  - `learning_rate`
  - `max_iter`
  - `l2_regularization`
  - `class_weight`

### 8. LogisticRegression

- 一手来源：
  - 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
- 核心优势：
  - 简单、可解释、报告友好
  - 多分类概率输出稳定
- 适合当前项目的原因：
  - 当前 baseline 并不弱，说明数据中有很强主效应
  - 作为非树型对照很有价值
- 潜在风险：
  - 上限通常不如强树模型
  - 需要依赖较好的编码与缩放
- 最小示例：

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    solver="saga",
    penalty="elasticnet",
    l1_ratio=0.2,
    C=2.0,
    class_weight="balanced",
    max_iter=5000,
    random_state=42,
)

model.fit(X_train_enc, y_train)
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `C`
  - `penalty`
  - `l1_ratio`
  - `solver`
  - `class_weight`

### 9. SVC (RBF)

- 一手来源：
  - 官方文档：<https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>
- 核心优势：
  - 对中小样本非线性问题有时很强
- 适合当前项目的原因：
  - 样本量还在 SVC 可承受范围内
  - 可作为“非树、非线性” classical 对照
- 潜在风险：
  - 调参敏感
  - 高维 OHE 后未必占优
  - 概率输出不是它的强项
- 最小示例：

```python
from sklearn.svm import SVC

model = SVC(
    kernel="rbf",
    C=5.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42,
)

model.fit(X_train_enc, y_train)
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `C`
  - `gamma`
  - `class_weight`

### 10. Oblique RF / SPORF / Rotation Forest

- 一手来源：
  - treeple 文档：<https://docs.neurodata.io/treeple/v0.6/modules/ensemble.html>
  - SPORF 官方站点：<https://neurodata.io/forests/>
  - Rotation Forest 论文条目：<https://pubmed.ncbi.nlm.nih.gov/16986543/>
- 核心优势：
  - 用线性组合特征做斜切分裂，可能更擅长复杂交互
- 适合当前项目的原因：
  - 如果你怀疑“学生状态”更依赖多变量组合阈值，而非单一特征阈值，这类方法值得探索
- 潜在风险：
  - Python 生态不如前三大 boosting 成熟
  - SPORF 官方站点明确写了 `not actively developed`
  - 不适合放在首轮正式实验主线
- 最小示例：

```python
from treeple import ObliqueRandomForestClassifier

model = ObliqueRandomForestClassifier(
    n_estimators=300,
    feature_combinations=2,
    random_state=42,
)

model.fit(X_train_enc, y_train)
pred = model.predict(X_valid)
```

- 优先可调参数：
  - `n_estimators`
  - `feature_combinations`
  - `max_depth`
  - `max_features`

## 最值得先做的优先级

建议优先顺序如下：

1. `CatBoostClassifier`
2. `LGBMClassifier`
3. `XGBClassifier`
4. `BalancedRandomForestClassifier`
5. `HistGradientBoostingClassifier`
6. `ExtraTreesClassifier` 继续系统调参，作为强锚点
7. `RandomForestClassifier` 继续系统调参，作为第二锚点
8. `LogisticRegression`
9. `SVC (RBF)`
10. `Oblique RF / SPORF / Rotation Forest`

## 最小实验建议

如果只允许先做一轮高收益实验，建议顺序是：

1. 先跑 `CatBoost / LightGBM / XGBoost`
2. 再补 `BalancedRandomForest`
3. 让所有新模型统一用：
   - 分层 CV
   - 主指标 `macro-F1`
   - 辅助指标每类 `recall`
4. 与当前 `ExtraTrees / RandomForest / LogisticRegression` 做并列比较

## 记录说明

- 本文优先引用官方文档、原始论文、官方站点。
- 其中 `Rotation Forest` 的主流 Python 官方实现不成熟，因此只建议作为探索项。
- 本文不代表这些模型一定优于现有 advanced models；它只说明哪些方向最值得先验证。
