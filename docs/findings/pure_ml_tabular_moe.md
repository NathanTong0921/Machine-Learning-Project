# Finding: Pure-ML Tabular MoE v1

## Summary

The first pure traditional-ML Tabular MoE implementation was tested with a GMM router and local LightGBM/CatBoost experts. It is feasible but does not beat the current global and ensemble baselines.

Best model:

| model | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gmm_moe_lightgbm_hard_k4_diag` | 0.7673 | 0.7129 | 0.8782 | 0.6821 | 0.7377 | 0.5168 | 0.8763 |

Current reference:

| model | accuracy | macro-F1 | AUC | enrolled acc |
| --- | ---: | ---: | ---: | ---: |
| LightGBM hist-column best | 0.7718 | 0.7196 | 0.8921 | 0.5336 |
| machine-only ensemble | 0.7696 | 0.7247 | 0.8982 | 0.5924 |
| final mixed OOF ensemble | 0.7824 | 0.7343 | 0.9013 | 0.5798 |

## Interpretation

The GMM router does not collapse: the best K=4 model has cluster shares from 0.1182 to 0.4008, and all clusters are large enough to train local experts.

The issue is not load balance. The issue is alignment. GMM routes by feature density, while the project metric depends on class discrimination, especially the ambiguous `Enrolled` class. The router finds dropout-heavy and graduate-heavy regions, but `Enrolled` remains mixed across clusters.

Therefore, the current result is a negative but useful finding: unsupervised manifold routing alone is not enough to outperform global tree models on this dataset.

## Recommended Next Step

If this line is continued, use a supervised error-predicting router:

1. Train candidate experts with out-of-fold predictions.
2. Label each training sample by the expert with the lowest OOF loss.
3. Train a router to predict that expert ID.
4. Aggregate with hard or soft expert routing.

This keeps the MoE idea but aligns routing with classification error instead of unsupervised feature density.

Detailed experiment record: `docs/search_tabular_moe.md`.
