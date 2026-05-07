# Pure-ML Tabular MoE Side Experiment

## Motivation

This side experiment tests a pure traditional-machine-learning mixture-of-experts idea for the student dropout task. The core question is whether a router can split the tabular feature manifold into local regions, then let local tree experts reduce the compromise made by one global model.

The implemented version is the easiest decoupled two-stage MoE:

1. Fit a router on `X` only.
2. Split the training set by router assignment.
3. Train one local expert per cluster.
4. Aggregate local expert probabilities with either hard or soft routing.

This is intentionally kept separate from the existing one-stage and two-stage decision-threshold scripts.

## Script And Outputs

Script:

- `scripts/ensemble/search_tabular_moe.py`

Outputs:

- `models/metrics/tabular_moe_metrics.csv`
- `models/metrics/tabular_moe_cluster_summary.csv`
- `models/metrics/tabular_moe_best.json`

## Implementation

Router:

- Preprocessing follows the current Appendix-aligned tabular pipeline: numeric features are median-imputed and standardized; categorical features are one-hot encoded.
- Router dense input dimension is 241: 19 numeric-scaled columns plus 222 one-hot categorical columns.
- PCA reduces the router input to 32 dimensions.
- `GaussianMixture` is used as the unsupervised router.
- Router grid:
  - `k`: 2, 3, 4, 5
  - covariance: `diag`, `full`

Experts:

- LightGBM local experts.
- CatBoost local experts.
- Experts are trained on raw 36-column data with native categorical handling where supported by the model wrapper.
- If a cluster is too small or lacks enough classes, the script falls back to a global expert for that cluster.

Aggregation:

- Hard routing: use the top-1 routed expert prediction.
- Soft routing: weighted average of expert probabilities using GMM posterior probabilities.

Compute:

- The run used `n_jobs=32`.
- Total searched configurations: 32.

## Best MoE Results

All metrics below are on the held-out test split used by the other project scripts.

| model | expert | routing | k | covariance | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gmm_moe_lightgbm_hard_k4_diag` | LightGBM | hard | 4 | diag | 0.7673 | 0.7129 | 0.8782 | 0.6821 | 0.7377 | 0.5168 | 0.8763 |
| `gmm_moe_lightgbm_soft_k4_diag` | LightGBM | soft | 4 | diag | 0.7666 | 0.7106 | 0.8825 | 0.6820 | 0.7424 | 0.5042 | 0.8763 |
| `gmm_moe_lightgbm_soft_k5_diag` | LightGBM | soft | 5 | diag | 0.7636 | 0.7062 | 0.8774 | 0.6795 | 0.7307 | 0.4958 | 0.8808 |
| `gmm_moe_catboost_hard_k2_full` | CatBoost | hard | 2 | full | 0.7485 | 0.7050 | 0.8830 | 0.6564 | 0.7166 | 0.5966 | 0.8235 |
| `gmm_moe_catboost_soft_k2_diag` | CatBoost | soft | 2 | diag | 0.7425 | 0.7010 | 0.8778 | 0.6449 | 0.6979 | 0.6092 | 0.8190 |

The best macro-F1 MoE is `gmm_moe_lightgbm_hard_k4_diag`.

## Comparison Against Current Baselines

| model | family | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gmm_moe_lightgbm_hard_k4_diag` | pure-ML MoE | 0.7673 | 0.7129 | 0.8782 | 0.7377 | 0.5168 | 0.8763 |
| LightGBM hist-column best | one-stage tree | 0.7718 | 0.7196 | 0.8921 | 0.7471 | 0.5336 | 0.8733 |
| tuned CatBoost | one-stage tree | 0.7590 | 0.7222 | 0.8959 | n/a | n/a | n/a |
| machine-only ensemble | one-stage ensemble | 0.7696 | 0.7247 | 0.8982 | 0.7330 | 0.5924 | 0.8567 |
| thermometer tuned MultiView MLP | deep tabular | 0.7824 | 0.7350 | 0.8903 | 0.7447 | 0.5798 | 0.8793 |
| final mixed OOF ensemble | mixed ensemble | 0.7824 | 0.7343 | 0.9013 | 0.7471 | 0.5798 | 0.8778 |

Relative to the strongest machine-only ensemble, the best pure-ML MoE is:

| comparison | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MoE minus machine-only ensemble | -0.0023 | -0.0118 | -0.0200 | +0.0047 | -0.0756 | +0.0196 |

Relative to LightGBM hist-column best, the best pure-ML MoE is:

| comparison | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MoE minus LightGBM hist-column best | -0.0045 | -0.0067 | -0.0139 | -0.0094 | -0.0168 | +0.0030 |

## Router And Cluster Observations

Best model `gmm_moe_lightgbm_hard_k4_diag` cluster load:

| cluster | n | load share | dropout | enrolled | graduate | enrolled share | mean router confidence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1053 | 0.3401 | 381 | 254 | 418 | 0.2412 | 0.9293 |
| 1 | 436 | 0.1408 | 342 | 35 | 59 | 0.0803 | 0.9740 |
| 2 | 366 | 0.1182 | 146 | 64 | 156 | 0.1749 | 0.9531 |
| 3 | 1241 | 0.4008 | 125 | 203 | 913 | 0.1636 | 0.9562 |

There is no severe expert collapse in this run. The smallest cluster still contains 366 training samples, and no cluster needed fallback to the global expert.

The router also finds interpretable rough regions:

- Cluster 1 is dropout-heavy: 342 dropout vs. 35 enrolled vs. 59 graduate.
- Cluster 3 is graduate-heavy: 913 graduate vs. 125 dropout vs. 203 enrolled.
- Cluster 0 contains the highest enrolled share, but it is still mixed.

However, the clusters are not clean enough for local experts to beat the global models. The enrolled class remains spread across clusters, and the local split reduces the amount of data available to each expert.

## Finding

The current decoupled unsupervised GMM-router MoE is a useful negative result:

- It validates that manifold routing is technically feasible in this project pipeline.
- It does not currently improve the leaderboard objective.
- The best MoE macro-F1 is 0.7129, below LightGBM hist-column best, tuned CatBoost, and the machine-only ensemble.
- The best MoE AUC is also weaker than the global tree baselines.
- The highest enrolled recall among the top MoE variants comes from CatBoost K=2, but that trades away overall macro-F1 and accuracy.

The likely reason is that an unsupervised router optimizes feature density, not downstream classification error. It can discover broad dropout-heavy and graduate-heavy regions, but it does not isolate the ambiguous enrolled samples well enough.

## Follow-Up Completed

The next MoE version was completed as a supervised error-predicting router:

1. Train several global experts with out-of-fold predictions.
2. For each training sample, compute which expert gives the lowest cross-entropy or correct-class loss.
3. Train a router to predict the best expert ID from features.
4. At inference, use either top-1 expert routing or soft expert weighting.

This is documented in `docs/search_supervised_error_router_moe.md`. The best version reaches test macro-F1 0.7318, improving over this GMM MoE result and the machine-only ensemble.
