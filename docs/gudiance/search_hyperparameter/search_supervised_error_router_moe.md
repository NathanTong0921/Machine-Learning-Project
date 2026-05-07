# Supervised Error-Predicting Router MoE

## Motivation

This is the second pure-ML Tabular MoE experiment after the GMM-router version.

The GMM router splits the feature manifold without using labels. It did not collapse, but it also did not align well with the final classification error. This experiment replaces the unsupervised router with a supervised error-predicting router:

1. Train candidate ML experts with out-of-fold predictions.
2. For each training row, compute the true-class negative log loss of every expert.
3. Assign a pseudo-label to the expert with the lowest OOF loss.
4. Train a router to predict that best-expert pseudo-label from the original features.
5. At inference, aggregate expert probabilities by hard or soft router output.

No deep learning, TabPFN, MLP, or transformer member is used.

## Script And Outputs

Script:

- `scripts/ensemble/search_supervised_error_router_moe.py`

Full 8-expert outputs:

- `models/metrics/supervised_error_router_moe_metrics.csv`
- `models/metrics/supervised_error_router_moe_best.json`

Core 3-expert outputs:

- `models/metrics/supervised_error_router_moe_core3_metrics.csv`
- `models/metrics/supervised_error_router_moe_core3_best.json`

Follow-up v2 / heterogeneous outputs:

- `models/metrics/supervised_error_router_moe_v2_metrics.csv`
- `models/metrics/supervised_error_router_moe_v2_best.json`
- `models/metrics/supervised_error_router_moe_hetero_metrics.csv`
- `models/metrics/supervised_error_router_moe_hetero_best.json`
- Detailed follow-up note: `docs/search_supervised_error_router_moe_v2.md`

Commands used:

```bash
python scripts/ensemble/search_supervised_error_router_moe.py --cv 5 --n-jobs 64
python scripts/ensemble/search_supervised_error_router_moe.py --quick --cv 5 --n-jobs 64 --output-prefix supervised_error_router_moe_core3
```

## Candidate Experts

Full version:

- CatBoost tuned
- LightGBM hist-column best
- LightGBM tuned
- GBDT log-loss
- HistGradientBoosting
- Extra Trees
- Bagging Decision Tree
- XGBoost hist-column best

Core3 version:

- CatBoost tuned
- LightGBM hist-column best
- GBDT log-loss

Router models tested in the full run:

- Logistic Regression router
- Extra Trees router
- Random Forest router

Pseudo-label modes tested in the full run:

- `raw_loss`: choose the expert with the lowest raw OOF true-class loss.
- `centered_loss`: subtract each expert's average loss before choosing.
- `zscore_loss`: z-score each expert's loss before choosing.

The `centered_loss` and `zscore_loss` variants were included because supervised routers can collapse toward the strongest global expert. They are a simple load-balancing guard: the router target is based on relative per-expert advantage, not only absolute loss.

## Best Results

All metrics below are on the held-out test split.

| model | expert set | router | routing | pseudo-label | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| supervised error-router MoE | core3 | Logistic Regression | soft | raw loss | 0.7756 | 0.7318 | 0.8979 | 0.6867 | 0.7400 | 0.6092 | 0.8582 |
| supervised error-router MoE | full8 | Logistic Regression | soft | raw loss | 0.7756 | 0.7268 | 0.8986 | 0.6882 | 0.7330 | 0.5630 | 0.8793 |
| supervised error-router MoE | full8 | Logistic Regression | soft | centered loss | 0.7756 | 0.7255 | 0.8968 | 0.6890 | 0.7447 | 0.5504 | 0.8763 |
| supervised error-router MoE | full8 | Random Forest | soft | z-score loss | 0.7786 | 0.7236 | 0.8976 | 0.6962 | 0.7447 | 0.5168 | 0.8944 |
| supervised error-router MoE | core3 | Logistic Regression | hard | raw loss | 0.7651 | 0.7202 | 0.8957 | 0.6753 | 0.7307 | 0.5966 | 0.8477 |

The best supervised MoE is the core3 soft-routed logistic-router version.

## Comparison Against Existing Results

| model | family | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GMM MoE best | unsupervised pure-ML MoE | 0.7673 | 0.7129 | 0.8782 | 0.7377 | 0.5168 | 0.8763 |
| LightGBM hist-column best | one-stage tree | 0.7718 | 0.7196 | 0.8921 | 0.7471 | 0.5336 | 0.8733 |
| machine-only ensemble | one-stage ML ensemble | 0.7696 | 0.7247 | 0.8982 | 0.7330 | 0.5924 | 0.8567 |
| supervised error-router MoE core3 | pure-ML MoE | 0.7756 | 0.7318 | 0.8979 | 0.7400 | 0.6092 | 0.8582 |
| final mixed OOF ensemble | mixed ensemble | 0.7824 | 0.7343 | 0.9013 | 0.7471 | 0.5798 | 0.8778 |
| thermometer tuned MultiView MLP | deep tabular | 0.7824 | 0.7350 | 0.8903 | 0.7447 | 0.5798 | 0.8793 |

Relative to the machine-only ensemble:

| comparison | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| supervised MoE core3 minus machine-only ensemble | +0.0060 | +0.0070 | -0.0003 | +0.0070 | +0.0168 | +0.0015 |

Relative to the unsupervised GMM MoE:

| comparison | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| supervised MoE core3 minus GMM MoE | +0.0083 | +0.0189 | +0.0197 | +0.0023 | +0.0924 | -0.0181 |

## Router Collapse Check

Full 8-expert raw-loss pseudo-label distribution:

| expert | pseudo-label share |
| --- | ---: |
| XGBoost hist-column best | 0.5275 |
| LightGBM tuned | 0.1315 |
| GBDT log-loss | 0.0827 |
| HistGradientBoosting | 0.0827 |
| Extra Trees | 0.0501 |
| CatBoost tuned | 0.0484 |
| Bagging Decision Tree | 0.0407 |
| LightGBM hist-column best | 0.0365 |

Full 8-expert logistic router hard route share on test:

| expert | test route share |
| --- | ---: |
| XGBoost hist-column best | 0.2327 |
| LightGBM hist-column best | 0.1431 |
| HistGradientBoosting | 0.1205 |
| LightGBM tuned | 0.1175 |
| CatBoost tuned | 0.1145 |
| Bagging Decision Tree | 0.1032 |
| Extra Trees | 0.0926 |
| GBDT log-loss | 0.0761 |

Core3 raw-loss pseudo-label distribution:

| expert | pseudo-label share |
| --- | ---: |
| LightGBM hist-column best | 0.5420 |
| GBDT log-loss | 0.3052 |
| CatBoost tuned | 0.1528 |

Core3 logistic router hard route share on test:

| expert | test route share |
| --- | ---: |
| LightGBM hist-column best | 0.4639 |
| CatBoost tuned | 0.2937 |
| GBDT log-loss | 0.2425 |

There is no total collapse to a single expert, but the full 8-expert pseudo-labels are clearly skewed toward XGBoost. The router soft probabilities reduce the hard assignment skew, which is why soft routing consistently beats hard routing.

The core3 setup is more stable: it keeps the strongest LightGBM expert dominant but still assigns meaningful mass to CatBoost and GBDT. This is also the best macro-F1 result.

## Finding

The supervised error-predicting router is a real improvement over the unsupervised GMM MoE and is now the best pure-ML MoE variant.

The main practical finding is that MoE quality depends more on expert complementarity than expert count. Adding many weaker or highly correlated experts makes the pseudo-label space noisier and does not improve the final metric. A small, diverse expert pool plus soft logistic routing works best so far.

For the report, this is a stronger side story than GMM MoE:

- GMM MoE: useful negative baseline; feature-density routing is not enough.
- Supervised error-router MoE: positive result; OOF loss routing improves pure ML performance and Enrolled recall.
- Remaining gap: it is still slightly below the best mixed ensemble and tuned MLP macro-F1.

## Follow-Up v2 Result

Four improvement ideas were tested after this result:

- Router-only meta features.
- Margin-weighted router training.
- One-step M-step specialist retraining.
- Heterogeneous PCA-fused one-hot ElasticNet / calibrated LinearSVC experts.

The best macro-F1 did not improve beyond the core3 result. The most useful side effect is that margin-weighted routing can push `Enrolled` accuracy up to 0.6303, but it sacrifices `Graduate` accuracy enough that macro-F1 remains lower.

The follow-up is documented in `docs/search_supervised_error_router_moe_v2.md`.
