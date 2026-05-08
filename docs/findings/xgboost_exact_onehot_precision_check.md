# XGBoost Exact One-Hot Precision Check

Date: 2026-05-08

## Scope

This note records the follow-up check after the histogram-column experiment:

- XGBoost was rerun with `tree_method="exact"` and one-hot encoded categorical features.
- LightGBM was also checked with higher `max_bin` and `use_quantile_bins=False`, but this is still histogram-based; LightGBM does not provide a true exact-split equivalent in this setup.

Script:

- `scripts/boost/compare_xgb_lgb_non_hist_precision.py`

Saved outputs:

- `models/metrics/boost/xgb_lgb_non_hist_precision_metrics.csv`
- `models/metrics/boost/xgboost_exact_onehot_search_grid.csv`
- `models/metrics/boost/xgboost_exact_onehot_best_params.json`
- `models/metrics/boost/lightgbm_quantile_off_search_grid.csv`
- `models/metrics/boost/lightgbm_quantile_off_best_params.json`

## Main Result

| model | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGB exact one-hot, original structural params | 0.7801 | 0.7180 | 0.8939 | 0.7635 | 0.4580 | 0.9065 |
| XGB exact one-hot, 10-fold local tuned | 0.7764 | 0.7127 | 0.8974 | 0.7518 | 0.4496 | 0.9095 |
| XGB hist replay | 0.7696 | 0.6642 | 0.8887 | 0.7939 | 0.2479 | 0.9412 |

## Best 10-Fold Local Search Params

```json
{
  "colsample_bytree": 0.95,
  "learning_rate": 0.05,
  "max_depth": 5,
  "min_child_weight": 1,
  "n_estimators": 350,
  "reg_lambda": 1.0,
  "subsample": 0.9
}
```

10-fold CV macro-F1 was `0.7160 +/- 0.0169`, but held-out test macro-F1 was only `0.7127`.

## Finding

XGBoost exact + one-hot fixes much of the `Enrolled` collapse seen in native categorical histogram XGBoost, but local hyperparameter search did not beat the original exact structural params on held-out macro-F1.

For report use, keep XGB exact one-hot as a useful diagnostic baseline, not as the main model. The best XGB exact result remains the original structural setting:

```json
{
  "max_depth": 6,
  "learning_rate": 0.05,
  "n_estimators": 400,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "reg_alpha": 0.0,
  "reg_lambda": 1.0
}
```

LightGBM results in this script should be described as high-precision / quantile-off histogram checks, not as non-histogram LightGBM.
