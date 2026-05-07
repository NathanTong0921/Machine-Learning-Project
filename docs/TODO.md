# TODO and Findings

## Current Findings

- Current best macro-F1 is `0.73425`, from the OOF probability ensemble with added semester-feature tree members.
- `LightGBM tuned` finished its 10-fold grid search with CV macro-F1 `0.72764`, but held-out test macro-F1 was only `0.70423`.
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
- Class-wise semester means show a strong monotonic academic-performance gap:
  - Dropout: `2nd sem approved = 1.94`, `2nd sem grade = 5.90`
  - Enrolled: `2nd sem approved = 4.06`, `2nd sem grade = 11.12`
  - Graduate: `2nd sem approved = 6.18`, `2nd sem grade = 12.70`
- PCA is weak as direct class-separation evidence:
  - `PC1 explained variance = 0.2340`
  - `PC2 explained variance = 0.0974`
  - `PC1 + PC2 = 0.3314`
  - `PCA silhouette by target = 0.0032`
- This supports using t-SNE only as local-neighborhood visualization, not as standalone classification evidence.
- TargetEncoder + semester ratio features was tested:
  - 5-fold best macro-F1 mean: `targetenc_semester_logistic = 0.71311`, `targetenc_semester_histgb = 0.71186`
  - final test best macro-F1: `targetenc_semester_histgb = 0.71970`
  - `targetenc_semester_logistic` gives high `Enrolled recall = 0.65966` on test but lower global macro-F1 `0.70439`
  - Conclusion: useful supporting baseline and good Enrolled-recall evidence, but not better than current CatBoost/MultiView leaders.

## Encoding Findings

- Keep CatBoost categorical features native. CatBoost internally handles categorical features with ordered target statistics and categorical combinations, so manual one-hot is not recommended.
- Keep LightGBM categorical features native with categorical columns encoded as pandas `category` or non-negative integer categories.
- For sklearn baselines, try `TargetEncoder` with internal cross-fitting. This is especially relevant for categorical fields like `Course`, `Application mode`, `Previous qualification`, and parent occupation fields.
- Avoid leakage: target encoding must be fit inside CV folds or via `TargetEncoder.fit_transform` cross-fitting on training folds only.
- For neural tabular models, an entity-embedding version may be useful, but it is a larger change than target-encoded sklearn baselines.

## Priority TODO

### P0. Finish Tuned LightGBM

Completed:

- Best CV macro-F1: `0.72764`
- Best params: `learning_rate=0.03`, `n_estimators=300`, `num_leaves=63`, `min_child_samples=40`, `max_depth=-1`, `colsample_bytree=1.0`, `subsample=0.8`
- Held-out test macro-F1: `0.70423`
- Held-out test Enrolled recall/F1: `0.50000 / 0.49174`

Refresh summaries when producing final tables:

```bash
python scripts/reporting/summarize_results.py
python scripts/reporting/summarize_primary_contenders.py
```

Conclusion:

- Do not use LightGBM tuned as the primary single model.
- Keep it only as a candidate ensemble member or supporting negative result.
- Adding semester ratio/delta features rescues LightGBM substantially:
  - `lightgbm_tuned`: test macro-F1 `0.70423`
  - `lightgbm_tuned_semester`: test macro-F1 `0.72142`

### P0. TargetEncoder + Semester Ratio Features

Implemented a lightweight feature-encoding experiment:

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

Current result:

- It did not exceed the best models.
- It remains useful in the report because it validates that semester-ratio features and target encoding are competitive, interpretable baselines.

### P1. OOF Probability Ensemble

Build an out-of-fold probability ensemble:

- `catboost_tuned`
- `lightgbm_tuned`
- `multiview_gated_mlp_best`
- `tabpfn`

Start with weighted averaging, then try a simple multinomial logistic meta-learner.

Rules:

- Tune weights only on CV/OOF predictions.
- Do not tune on the final test set.

Current result with LightGBM candidate:

- Members: `catboost_tuned`, `lightgbm_tuned`, `multiview_gated_mlp_best`, `tabpfn`
- Best OOF weights:
  - `catboost_tuned = 0.50`
  - `lightgbm_tuned = 0.00`
  - `multiview_gated_mlp_best = 0.05`
  - `tabpfn = 0.45`
- OOF macro-F1: `0.73134`
- Test macro-F1: `0.73203`
- Test accuracy: `0.78690`
- Test AUC: `0.90406`
- Test Enrolled recall/F1: `0.53782 / 0.53782`

Conclusion:

- This is currently the strongest result by macro-F1.
- LightGBM was tested inside the ensemble but received zero weight, so it does not improve the ensemble.

Updated ensemble with semester-feature tree members:

- Added candidates:
  - `catboost_tuned_semester`
  - `lightgbm_tuned_semester`
- Best OOF weights:
  - `catboost_tuned = 0.00`
  - `lightgbm_tuned = 0.30`
  - `multiview_gated_mlp_best = 0.00`
  - `tabpfn = 0.20`
  - `catboost_tuned_semester = 0.50`
  - `lightgbm_tuned_semester = 0.00`
- OOF macro-F1: `0.73889`
- Test macro-F1: `0.73425`
- Test accuracy: `0.78238`
- Test AUC: `0.90127`
- Test Enrolled recall/F1: `0.57983 / 0.54871`

Conclusion:

- This is the new strongest macro-F1 result.
- Semester features are useful mostly through ensemble diversity, not as a standalone CatBoost replacement.
- The raw LightGBM member receives weight despite weak standalone test macro-F1, suggesting it contributes complementary probability structure in the ensemble.

### P1. Decision Rule / Threshold Optimization

Because TabPFN has high AUC but lower macro-F1, try class-specific probability biases or thresholds:

- Optimize class bias terms on validation/OOF predictions.
- Track whether `Enrolled` recall improves without excessive macro-F1 loss.
- Do not optimize directly on the test set.

CatBoost tuned decision-bias experiment was run with 5-fold OOF probabilities:

- Best OOF log-probability bias: `Dropout=0.0`, `Enrolled=0.05`, `Graduate=0.25`
- OOF macro-F1 improved from `0.71257` to `0.71760`
- Test macro-F1 changed from `0.72222` to `0.72165`
- Test accuracy improved from `0.75904` to `0.76581`
- Test AUC improved from `0.89594` to `0.89730`
- Test Enrolled recall dropped from `0.65546` to `0.60504`

Conclusion:

- Bias optimization helps overall accuracy/AUC and OOF macro-F1, but it does not improve final test macro-F1 or Enrolled recall for CatBoost.
- Keep raw CatBoost tuned as the main CatBoost result unless the report emphasizes accuracy/AUC instead of macro-F1.

OOF ensemble decision-bias experiment was run on cached ensemble probabilities:

- Best OOF log-probability bias: `Dropout=0.0`, `Enrolled=0.0`, `Graduate=-0.1`
- OOF macro-F1 improved from `0.73134` to `0.73380`
- Test macro-F1 changed from `0.73203` to `0.73017`
- Test accuracy changed from `0.78690` to `0.78087`
- Test Enrolled recall/F1 changed from `0.53782 / 0.53782` to `0.56303 / 0.54032`

Conclusion:

- Ensemble bias improves Enrolled recall/F1 slightly on test but reduces global macro-F1 and accuracy.
- Keep raw OOF ensemble as the main best model for macro-F1.
- Bias-adjusted ensemble can be reported only as an Enrolled-recall tradeoff variant.

OOF ensemble decision-bias was also tested on the new semester-members ensemble:

- Best OOF log-probability bias: `Dropout=0.0`, `Enrolled=0.0`, `Graduate=-0.1`
- OOF macro-F1 improved from `0.73889` to `0.73923`
- Test macro-F1 dropped from `0.73425` to `0.72866`
- Test Enrolled recall/F1 changed from `0.57983 / 0.54871` to `0.58824 / 0.54054`

Conclusion:

- Do not apply bias to the new best ensemble.
- Bias tuning is useful diagnostic evidence but not a performance improvement for final macro-F1.

### P1. OOF Probability Meta-Learner

Regularized multinomial logistic stacking was tested on OOF probability features:

- Best meta-CV config:
  - feature set: member probabilities only
  - `C = 0.1`
  - `class_weight = balanced`
- Meta-CV macro-F1: `0.73097 ± 0.01182`
- Test macro-F1: `0.72257`
- Test Enrolled recall/F1: `0.65546 / 0.53793`

Conclusion:

- The meta-learner increases Enrolled recall, but hurts global macro-F1.
- Do not use it as the primary model.
- Keep it as evidence that class-balanced stacking over-corrects toward the minority Enrolled class.

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
