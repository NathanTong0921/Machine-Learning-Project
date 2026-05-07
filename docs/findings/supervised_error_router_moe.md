# Finding: Supervised Error-Predicting Router MoE

## Summary

Replacing the unsupervised GMM router with an OOF error-predicting router substantially improves the pure-ML MoE line.

Best result:

| model | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core3 supervised error-router MoE | 0.7756 | 0.7318 | 0.8979 | 0.6867 | 0.7400 | 0.6092 | 0.8582 |

This beats the previous GMM MoE and the machine-only ensemble on accuracy, macro-F1, dropout acc, enrolled acc, and graduate acc. It is still slightly below the final mixed ensemble and the tuned thermometer MultiView MLP on macro-F1.

## Why It Works

The router target is created from OOF expert losses rather than in-sample losses, so the router learns which expert is genuinely more reliable for a region instead of learning training-set overfit.

Soft routing works better than hard routing because it avoids brittle one-expert decisions near region boundaries.

## Collapse Check

Supervised routers can collapse if the pseudo-labels all point to the strongest expert. This did not fully happen, but the full 8-expert setup was skewed:

- Full8 raw-loss pseudo-labels gave 52.75% of samples to XGBoost hist-column best.
- Core3 raw-loss pseudo-labels gave 54.20% to LightGBM hist-column best, 30.52% to GBDT, and 15.28% to CatBoost.

The core3 version performs better, suggesting that expert complementarity matters more than the number of experts.

## Recommendation

Use this as the positive MoE side result:

- GMM MoE is the negative control.
- Supervised error-router MoE is the improved pure-ML MoE.
- The current best variant is core3 + Logistic Regression router + soft routing + raw OOF loss pseudo-labels.

Detailed record: `docs/search_supervised_error_router_moe.md`.

## Follow-Up v2

Additional attempts did not beat the core3 result:

| variant | best macro-F1 | note |
| --- | ---: | --- |
| router meta features | 0.7299 | Improves uncertainty awareness but not macro-F1. |
| margin-weighted router | 0.7290 | Raises `Enrolled` acc up to 0.6303 but hurts `Graduate`. |
| one-step M-step expert retraining | 0.7241 | Router weights were not clean enough for stable specialization. |
| ElasticNet / LinearSVC hetero expert | 0.7182 | Truly different boundary family, but weaker than GBDT as third expert. |

ElasticNet is still useful diagnostically: it reaches very high `Enrolled` recall as a single expert, but collapses `Graduate` recall. This suggests it should be treated as a constrained `Enrolled` specialist, not as a general third expert replacement.

Follow-up record: `docs/search_supervised_error_router_moe_v2.md`.
