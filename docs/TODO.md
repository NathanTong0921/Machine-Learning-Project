# TODO and Findings

## Current Findings

- Current best macro-F1 is still around `0.722`, led by `catboost_tuned` and `multiview_gated_mlp_best`.
- `LightGBM tuned` is still worth finishing because partial CV already reached about `0.7234`.
- `Enrolled` is the hardest class. It is only about `17.95%` of the full dataset according to the external visualization analysis.
- Cost-sensitive MultiViewGatedMLP improves `Enrolled` recall/F1 but reduces global macro-F1.
- 5-fold CV confirms the same tradeoff:
  - baseline: macro-F1 mean `0.70485`, Enrolled recall mean `0.60417`, Enrolled F1 mean `0.50319`
  - Enrolled x1.15: macro-F1 mean `0.69983`, Enrolled recall mean `0.67069`, Enrolled F1 mean `0.51110`
  - Enrolled x1.30 + dropout: macro-F1 mean `0.69788`, Enrolled recall mean `0.67986`, Enrolled F1 mean `0.51037`
- Visualization shows the most target-related features are semester performance and finance/status features:
  - `Curricular units 2nd sem (approved)`
  - `Curricular units 2nd sem (grade)`
  - `Curricular units 1st sem (approved)`
  - `Curricular units 1st sem (grade)`
  - `Tuition fees up to date`
  - `Scholarship holder`
  - `Age at enrollment`
  - `Debtor`
- Because target was encoded as `0=Dropout`, `1=Enrolled`, `2=Graduate`, correlations are exploratory only and should not be described as causal evidence.

## Encoding Findings

- Keep CatBoost categorical features native. CatBoost internally handles categorical features with ordered target statistics and categorical combinations, so manual one-hot is not recommended.
- Keep LightGBM categorical features native with categorical columns encoded as pandas `category` or non-negative integer categories.
- For sklearn baselines, try `TargetEncoder` with internal cross-fitting. This is especially relevant for categorical fields like `Course`, `Application mode`, `Previous qualification`, and parent occupation fields.
- Avoid leakage: target encoding must be fit inside CV folds or via `TargetEncoder.fit_transform` cross-fitting on training folds only.
- For neural tabular models, an entity-embedding version may be useful, but it is a larger change than target-encoded sklearn baselines.

## Priority TODO

### P0. Finish Tuned LightGBM

- Let `scripts/tune_lightgbm_baseline.py` finish on the external CPU machine.
- Compare final `lightgbm_tuned_metrics.csv` against `catboost_tuned_metrics.csv`.
- Refresh summaries after completion:

```bash
python scripts/summarize_results.py
python scripts/summarize_primary_contenders.py
```

### P0. TargetEncoder + Semester Ratio Features

Implement a lightweight feature-encoding experiment:

- Add semester ratio features:
  - `1st_approved_rate = approved / enrolled`
  - `2nd_approved_rate = approved / enrolled`
  - `1st_eval_pass_rate = approved / evaluations`
  - `2nd_eval_pass_rate = approved / evaluations`
  - `grade_delta = 2nd_sem_grade - 1st_sem_grade`
  - `approved_delta = 2nd_sem_approved - 1st_sem_approved`
  - optional: `tuition_scholarship_interaction`
  - optional: `debtor_tuition_interaction`
- Compare these models with 5-fold Stratified CV:
  - `TargetEncoder + LogisticRegression`
  - `TargetEncoder + LinearSVC` or calibrated SVM if probabilities are needed
  - `TargetEncoder + HistGradientBoostingClassifier`
- Main metric: `macro-F1`.
- Secondary metrics:
  - `Enrolled recall`
  - `Enrolled F1`
  - `macro AUC` where probabilities are available

Expected value:

- This tests whether better categorical encoding plus domain-shaped semester features can match or exceed the current `0.722` macro-F1 region.
- It is cheaper than adding more Transformer variants.

### P1. OOF Probability Ensemble

After LightGBM tuning is complete, build an out-of-fold probability ensemble:

- `catboost_tuned`
- `lightgbm_tuned`
- `multiview_gated_mlp_best`
- `tabpfn`

Start with weighted averaging, then try a simple multinomial logistic meta-learner.

Rules:

- Tune weights only on CV/OOF predictions.
- Do not tune on the final test set.

### P1. Decision Rule / Threshold Optimization

Because TabPFN has high AUC but lower macro-F1, try class-specific probability biases or thresholds:

- Optimize class bias terms on validation/OOF predictions.
- Track whether `Enrolled` recall improves without excessive macro-F1 loss.
- Do not optimize directly on the test set.

### P2. Neural Categorical Embeddings

Try a categorical-embedding version of MultiViewGatedMLP:

- Keep numeric features scaled.
- Feed categorical columns through trainable embeddings instead of one-hot encoding.
- Compare against the current one-hot MultiViewGatedMLP.

Risk:

- More implementation work.
- Current evidence suggests tree/encoding experiments are more promising than more neural variants.

## Lower Priority

- Do not prioritize more FT-Transformer/viewwise/Trompt/piecewise variants unless needed for report completeness.
- Do not prioritize XGBoost unless its categorical preprocessing is redesigned; current result is weak.
- Consider `HistGradientBoostingClassifier` mainly as part of the target-encoding and semester-ratio experiment.
