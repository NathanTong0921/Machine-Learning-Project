# Feature Processing Flow

这份文档由 `scripts/reporting/document_feature_processing_flow.py` 自动生成，反映当前代码和 `data/processed/train.csv` / `data/processed/test.csv` 的最新状态。

## 数据入口

| artifact | rows | columns | feature columns | target |
| --- | --- | --- | --- | --- |
| `data/processed/train.csv` | 3096 | 37 | 36 | `Target` |
| `data/processed/test.csv` | 1328 | 37 | 36 | `Target` |

`scripts/preprocessing/prepare_data.py` 不做 one-hot，也不做标准化；它只读取 raw CSV、清理列名/`Target`、做 stratified train/test split 并保存 processed CSV。

## 当前语义列划分

当前共有 `17` 个 semantic categorical 特征：

- `Marital status`
- `Application mode`
- `Course`
- `Daytime/evening attendance`
- `Previous qualification`
- `Nacionality`
- `Mother's qualification`
- `Father's qualification`
- `Mother's occupation`
- `Father's occupation`
- `Displaced`
- `Educational special needs`
- `Debtor`
- `Tuition fees up to date`
- `Gender`
- `Scholarship holder`
- `International`

当前共有 `19` 个 numeric/ordinal/continuous 特征：

- `Application order`
- `Previous qualification (grade)`
- `Admission grade`
- `Age at enrollment`
- `Curricular units 1st sem (credited)`
- `Curricular units 1st sem (enrolled)`
- `Curricular units 1st sem (evaluations)`
- `Curricular units 1st sem (approved)`
- `Curricular units 1st sem (grade)`
- `Curricular units 1st sem (without evaluations)`
- `Curricular units 2nd sem (credited)`
- `Curricular units 2nd sem (enrolled)`
- `Curricular units 2nd sem (evaluations)`
- `Curricular units 2nd sem (approved)`
- `Curricular units 2nd sem (grade)`
- `Curricular units 2nd sem (without evaluations)`
- `Unemployment rate`
- `Inflation rate`
- `GDP`

注意：`Application order` 当前不在 `CATEGORICAL_COLUMNS` 中，因此按 numeric/ordinal 处理。这与 PDF Table 1 的 `Numeric/ordinal` 标注一致。

补充：对线性/距离/MLP 模型，把 `Application order` 作为单个 scaled scalar 会隐含等距 ordinal 假设。已在 `scripts/mlp/compare_application_order_encodings_multiview.py` 中测试 thermometer 和 OOF target encoding；进一步在 `scripts/mlp/search_application_order_encoding_multiview.py` 中调参后，thermometer encoding 超过 tuned scalar MultiView MLP 的 macro-F1，OOF target encoding 则给出更高 AUC 和 `Enrolled` recall。

## Categorical Unique Count

| feature | train unique | test unique | all unique |
| --- | --- | --- | --- |
| `Marital status` | 6 | 6 | 6 |
| `Application mode` | 18 | 14 | 18 |
| `Course` | 17 | 17 | 17 |
| `Daytime/evening attendance` | 2 | 2 | 2 |
| `Previous qualification` | 17 | 13 | 17 |
| `Nacionality` | 17 | 13 | 21 |
| `Mother's qualification` | 28 | 24 | 29 |
| `Father's qualification` | 32 | 24 | 34 |
| `Mother's occupation` | 31 | 27 | 32 |
| `Father's occupation` | 40 | 33 | 46 |
| `Displaced` | 2 | 2 | 2 |
| `Educational special needs` | 2 | 2 | 2 |
| `Debtor` | 2 | 2 | 2 |
| `Tuition fees up to date` | 2 | 2 | 2 |
| `Gender` | 2 | 2 | 2 |
| `Scholarship holder` | 2 | 2 | 2 |
| `International` | 2 | 2 | 2 |

这些 unique count 解释了 one-hot 维度。实际模型 fit 时只用训练 split fit encoder，因此 test 中未见过的类别由 `handle_unknown='ignore'` 或 native categorical 机制处理。

## Classical ML: sklearn Pipeline One-Hot

适用脚本/模型：`scripts/baseline/run_baselines.py`、`scripts/bagging/*`、`scripts/boost/search_classic_boosting_models.py`、`scripts/ensemble/tune_machine_only_ensemble.py` 中的 Logistic Regression / SVM / AdaBoost / GBDT / HistGradientBoosting / Bagging decision tree / Extra Trees 等 sklearn 成员。

处理方式：

- numeric features: `SimpleImputer(strategy='median') -> StandardScaler()`

- categorical features: `SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore')`

- 最后喂给 sklearn estimator 的是拼接后的 tabular matrix。

| part | dimension fitted on processed train |
| --- | --- |
| numeric scaled | 19 |
| categorical one-hot | 222 |
| total model input | 241 |

按列展开：

| categorical feature | one-hot dimension on processed train |
| --- | --- |
| `Marital status` | 6 |
| `Application mode` | 18 |
| `Course` | 17 |
| `Daytime/evening attendance` | 2 |
| `Previous qualification` | 17 |
| `Nacionality` | 17 |
| `Mother's qualification` | 28 |
| `Father's qualification` | 32 |
| `Mother's occupation` | 31 |
| `Father's occupation` | 40 |
| `Displaced` | 2 |
| `Educational special needs` | 2 |
| `Debtor` | 2 |
| `Tuition fees up to date` | 2 |
| `Gender` | 2 |
| `Scholarship holder` | 2 |
| `International` | 2 |
## Logistic Regression And AdaBoost Status

| question | current status | where |
| --- | --- | --- |
| 普通 Logistic Regression 是否有 | 有。作为 baseline 使用 one-hot + scaled numeric 后接 `LogisticRegression(class_weight='balanced')`；grid-search 也调过 `C`。 | `src/baselines.py`, `scripts/baseline/run_baselines.py`, `scripts/ensemble/search_and_ensemble_baselines.py` |
| 特征工程 Logistic Regression 是否有 | 有。包括 polynomial features、RBF Nystroem map、target encoding + semester features。 | `scripts/feature_engineering/run_feature_engineered_baselines.py`, `scripts/feature_engineering/target_encoding_semester_features.py` |
| Logistic Regression stacking 是否有 | 有。OOF probability stacking 用 `LogisticRegression(max_iter=5000, class_weight='balanced')` 作为 meta learner。 | `scripts/ensemble/compare_cross_seed_stacking.py`, `scripts/ensemble/stack_oof_probability_ensemble.py` |
| AdaBoost with Logistic Regression base estimator 是否有 | 没有。当前 AdaBoost 只用 shallow `DecisionTreeClassifier` / stump-style base estimator；没有跑 LogisticRegression 作为 AdaBoost base estimator。 | `src/baselines.py`, `scripts/boost/search_classic_boosting_models.py`, `scripts/ensemble/tune_machine_only_ensemble.py` |

说明：`gbdt_log_loss` 覆盖的是 log-loss gradient boosting / LogitBoost-style 对照，不是 `AdaBoostClassifier(estimator=LogisticRegression(...))`。

## Classical ML: Native Categorical Tree Models

| model family | representative scripts | categorical treatment | numeric treatment | final model input |
| --- | --- | --- | --- | --- |
| CatBoost | `scripts/boost/tune_catboost_baseline.py`, ensemble CatBoost members | `CATEGORICAL_COLUMNS` cast to `str`, passed via `cat_features` indices | raw numeric values, no sklearn scaling | 36-column DataFrame; CatBoost handles categorical internally |
| LightGBM | `scripts/boost/search_hist_column_sampling.py`, ensemble LightGBM members | `CATEGORICAL_COLUMNS` cast to pandas `category`, passed via `categorical_feature` | raw numeric values, no sklearn scaling | 36-column DataFrame; LightGBM handles categorical splits internally |
| XGBoost hist | `scripts/boost/search_hist_column_sampling.py`, ensemble XGBoost hist members | `CATEGORICAL_COLUMNS` cast to pandas `category`, `enable_categorical=True` | raw numeric values, no sklearn scaling | 36-column DataFrame; XGBoost hist handles categorical features natively |

这里没有 one-hot 展开；`max_bin`、`column_fraction` 等参数作用在原始 36 列特征空间上。

## Advanced DL: MultiView Gated MLP / TabM-Style MLP

适用脚本：`scripts/mlp/run_advanced.py`、`scripts/mlp/retrain_multiview_current_preprocess.py`、`scripts/mlp/search_multiview_current_preprocess.py`、`scripts/mlp/compare_tabm_style_models.py` 等。

处理方式：每个 view 单独 fit 一个 `ColumnTransformer`。下表使用 reference seed `62`，也就是当前 current-preprocess 搜索和 best checkpoint 的主要口径。

- view 内 numeric: median impute + standard scaling

- view 内 categorical: most-frequent impute + dense one-hot

- 最后喂给模型的是多个 tensor：`enrollment_profile`, `family_finance`, `first_semester`, `second_semester_macro`

| view | input dim | # categorical cols | categorical cols | # numeric cols | numeric cols |
| --- | --- | --- | --- | --- | --- |
| `enrollment_profile` | 80 | 7 | `Marital status`, `Application mode`, `Course`, `Daytime/evening attendance`, `Previous qualification`, `Nacionality`, `International` | 4 | `Application order`, `Previous qualification (grade)`, `Admission grade`, `Age at enrollment` |
| `family_finance` | 133 | 10 | `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation`, `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder` | 0 | - |
| `first_semester` | 6 | 0 | - | 6 | `Curricular units 1st sem (credited)`, `Curricular units 1st sem (enrolled)`, `Curricular units 1st sem (evaluations)`, `Curricular units 1st sem (approved)`, `Curricular units 1st sem (grade)`, `Curricular units 1st sem (without evaluations)` |
| `second_semester_macro` | 9 | 0 | - | 9 | `Curricular units 2nd sem (credited)`, `Curricular units 2nd sem (enrolled)`, `Curricular units 2nd sem (evaluations)`, `Curricular units 2nd sem (approved)`, `Curricular units 2nd sem (grade)`, `Curricular units 2nd sem (without evaluations)`, `Unemployment rate`, `Inflation rate`, `GDP` |

在 reference seed `62` 下，当前 MultiView total dim 是 `228`。其中 `enrollment_profile=80` 只是单个 view，不是全量 one-hot 宽度。旧 checkpoint 的 `enrollment_profile=87` 来自把 `Application order` 当 categorical，旧兼容 total dim 是 `235`。

注意：`prepare_multiview_data()` 会先把 processed train 再切 internal train/val，并只在 internal train 上 fit one-hot encoder；所以 view dims 依赖 seed/fold。例如 default seed `42` 不一定等于 reference seed `62` 的 `228`。

## Advanced DL: FTTransformer / View-Wise / Piecewise / Trompt-Style Transformer

适用脚本：`scripts/transformer/compare_advanced_models.py`、`compare_viewwise_gated_transformers.py`、`compare_piecewise_numerical_models.py`、`compare_trompt_style_models.py` 等。

处理方式：

- categorical features 不做 one-hot；按训练 split 建立 category -> integer id，未知类别映射到 0。

- numeric features 做 median fill，然后按训练 split mean/std 标准化。

- 最后喂给模型两个 tensor：`x_cat` 和 `x_num`。

- Piecewise FT 的 numeric 输入仍是标准化后的 `x_num`，但分箱边界由原始 internal train numeric values 计算；不要把它描述成在标准化值上直接分箱。

| tensor | shape per sample | meaning |
| --- | --- | --- |
| `x_cat` | 17 | 17 个 categorical id，进入 embedding/tokenizer |
| `x_num` | 19 | 19 个标准化 numeric/ordinal/continuous value |
categorical cardinalities 基于 internal train split 统计，含 unknown id 0：

| categorical feature | cardinality incl. unknown |
| --- | --- |
| `Marital status` | 7 |
| `Application mode` | 18 |
| `Course` | 18 |
| `Daytime/evening attendance` | 3 |
| `Previous qualification` | 18 |
| `Nacionality` | 16 |
| `Mother's qualification` | 27 |
| `Father's qualification` | 28 |
| `Mother's occupation` | 32 |
| `Father's occupation` | 38 |
| `Displaced` | 3 |
| `Educational special needs` | 3 |
| `Debtor` | 3 |
| `Tuition fees up to date` | 3 |
| `Gender` | 3 |
| `Scholarship holder` | 3 |
| `International` | 3 |
## TabPFN

适用脚本：`scripts/transformer/run_tabpfn_baseline.py`。当前代码没有显式 one-hot、scaling 或 categorical cast；直接把 36-column pandas DataFrame 传入 `TabPFNClassifier.fit(X_train, y_train)`。因此：

- categorical integer codes 仍作为原始列值进入 TabPFN。

- numeric/ordinal/continuous features 也作为原始列值进入 TabPFN。

- 具体内部归一化/编码由 TabPFN 包自身处理，项目代码层面没有手工展开。

## Summary: What Is Fed Into Each Model

| branch | one-hot? | scaled numeric? | native categorical? | final input |
| --- | --- | --- | --- | --- |
| sklearn ML baselines/bagging/classic boosting | yes, 17 categorical columns | yes | no | 241-dim transformed matrix |
| CatBoost | no | no | yes, `str` cat columns | raw 36-column DataFrame + cat feature indices |
| LightGBM | no | no | yes, pandas `category` | raw 36-column DataFrame + categorical_feature names |
| XGBoost hist | no | no | yes, pandas `category` | raw 36-column DataFrame with `enable_categorical=True` |
| MultiView Gated MLP / TabM-style | yes, within each view | yes, within each view | no | four dense view tensors, seed 62 total dim 228 |
| FTTransformer-style advanced models | no | yes | embedding ids | `x_cat` 17 ids + `x_num` 19 floats |
| TabPFN | no explicit project-side one-hot | no explicit project-side scaling | not explicitly declared | raw 36-column DataFrame |
## Source Files To Check

- `src/config.py`: semantic categorical columns and MultiView feature groups

- `src/preprocessing.py`: sklearn one-hot preprocessing

- `src/advanced_models.py`: MultiView and FT-style preprocessing

- `scripts/boost/tune_catboost_baseline.py`: CatBoost native categorical path

- `scripts/boost/search_hist_column_sampling.py`: LightGBM/XGBoost native categorical path

- `scripts/ensemble/tune_machine_only_ensemble.py`: mixed machine-only ensemble preprocessing

- `scripts/transformer/run_tabpfn_baseline.py`: raw DataFrame TabPFN input

