# Supervised Error-Router MoE v2 Follow-Up

## Goal

The best pure-ML MoE was close to the tuned thermometer MultiView MLP:

| model | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| core3 supervised error-router MoE | 0.7756 | 0.7318 | 0.8979 | 0.7400 | 0.6092 | 0.8582 |
| thermometer tuned MultiView MLP | 0.7824 | 0.7350 | 0.8903 | 0.7447 | 0.5798 | 0.8793 |

This follow-up tests four possible ways to close the remaining macro-F1 gap:

1. Router-only meta features.
2. Margin-weighted router training.
3. One M-step specialist expert retraining.
4. Heterogeneous linear / LinearSVC experts with one-hot and PCA-fused curricular features.

## Scripts And Outputs

Scripts:

- `scripts/ensemble/search_supervised_error_router_moe_v2.py`
- `scripts/ensemble/search_supervised_error_router_moe_hetero.py`

Outputs:

- `models/metrics/supervised_error_router_moe_v2_metrics.csv`
- `models/metrics/supervised_error_router_moe_v2_best.json`
- `models/metrics/supervised_error_router_moe_hetero_metrics.csv`
- `models/metrics/supervised_error_router_moe_hetero_best.json`

Commands:

```bash
python scripts/ensemble/search_supervised_error_router_moe_v2.py --cv 5 --n-jobs 64
python scripts/ensemble/search_supervised_error_router_moe_hetero.py --cv 5 --n-jobs 64
```

## v2 Router Search

Tested router feature variants:

- `raw`: original tabular features.
- `thermometer`: original features plus `Application order >= 1..9`.
- `meta`: thermometer features plus expert probability entropy, confidence, mean probability, probability standard deviation, and disagreement features.

Tested router models:

- Logistic Regression router
- Extra Trees router
- Random Forest router

Tested margin-weight modes:

- none
- raw margin
- sqrt margin
- log margin

Best v2 results:

| model | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw logistic router, no margin, soft | 0.7756 | 0.7318 | 0.8979 | 0.6867 | 0.7400 | 0.6092 | 0.8582 |
| thermometer logistic router, no margin, soft | 0.7741 | 0.7302 | 0.8981 | 0.6850 | 0.7354 | 0.6092 | 0.8582 |
| meta random forest router, sqrt margin, soft | 0.7711 | 0.7299 | 0.8987 | 0.6777 | 0.7330 | 0.6261 | 0.8477 |
| raw extra trees router, sqrt margin, soft | 0.7726 | 0.7299 | 0.8976 | 0.6827 | 0.7377 | 0.6176 | 0.8507 |
| raw logistic router, sqrt margin, soft | 0.7703 | 0.7290 | 0.8970 | 0.6798 | 0.7400 | 0.6303 | 0.8401 |

Finding:

- Meta features and margin weighting help the router recover more `Enrolled` samples.
- The best `Enrolled` accuracy rises from 0.6092 to 0.6303.
- This comes at a graduate-recall cost, so macro-F1 does not improve beyond 0.7318.

## M-Step Specialist Retraining

The best v2 soft router was used to refit the three core experts with router soft probabilities as `sample_weight`.

| model | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M-step soft | 0.7711 | 0.7241 | 0.8987 | 0.6812 | 0.7447 | 0.5714 | 0.8597 |
| M-step hard | 0.7568 | 0.7056 | 0.8948 | 0.6653 | 0.7494 | 0.5210 | 0.8462 |

Finding:

- One-step specialist retraining did not help.
- The likely reason is that the router soft weights are not sharp enough to define clean local expert regions.
- Reweighting reduces the experts' global calibration and hurts the ambiguous `Enrolled` class.

## Heterogeneous Linear / LinearSVC Expert Search

The heterogeneous experiment keeps:

- CatBoost tuned
- LightGBM hist-column best

Then it replaces GBDT with one of:

- GBDT log-loss control
- PCA-fused thermometer ElasticNet Logistic Regression
- PCA-fused thermometer calibrated LinearSVC

Important preprocessing detail:

- Linear / SVM experts use numeric standardization plus categorical one-hot.
- `Application order` is augmented with thermometer features.
- The 12 `Curricular units 1st/2nd sem (...)` grade/credit features are standardized and compressed with `PCA(n_components=2)` before entering the linear/SVM expert.
- RBF SVM was deliberately not used in the final run because high-dimensional one-hot tabular spaces make RBF distances less reliable and slower.

Best heterogeneous results:

| model | third expert | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core3 control, raw logistic router, soft | GBDT | 0.7756 | 0.7318 | 0.8979 | 0.6867 | 0.7400 | 0.6092 | 0.8582 |
| core3 control, sqrt-margin router, soft | GBDT | 0.7703 | 0.7290 | 0.8970 | 0.6798 | 0.7400 | 0.6303 | 0.8401 |
| LinearSVC C=0.5, meta router, soft | LinearSVC | 0.7636 | 0.7182 | 0.8911 | 0.6717 | 0.7471 | 0.5798 | 0.8401 |
| LinearSVC C=1.0, raw router, soft | LinearSVC | 0.7620 | 0.7170 | 0.8894 | 0.6730 | 0.7283 | 0.5924 | 0.8446 |
| ElasticNet C=0.5 l1=0.2, raw router, soft | ElasticNet | 0.7425 | 0.7116 | 0.8918 | n/a | 0.7237 | 0.6849 | 0.7753 |

Single linear expert behavior:

| expert | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LinearSVC C=0.5 | 0.6777 | 0.6412 | 0.8218 | 0.7096 | 0.5210 | 0.7134 |
| LinearSVC C=1.0 | 0.6777 | 0.6408 | 0.8195 | 0.7049 | 0.5210 | 0.7164 |
| ElasticNet C=0.5 l1=0.2 | 0.5678 | 0.5778 | 0.8364 | 0.6230 | 0.8361 | 0.4359 |
| ElasticNet C=0.2 l1=0.5 | 0.5497 | 0.5630 | 0.8373 | 0.5972 | 0.8529 | 0.4103 |

Finding:

- Linear / ElasticNet experts are genuinely heterogeneous but not strong enough as direct replacements for GBDT.
- ElasticNet heavily favors `Enrolled`, reaching high `Enrolled` recall by sacrificing `Graduate` recall.
- LinearSVC is more balanced than ElasticNet but still below GBDT in macro-F1.
- The router cannot fully convert this heterogeneity into a macro-F1 gain.

## Overall Conclusion

The 0.7350 MLP threshold was not crossed.

Best pure-ML MoE remains:

| model | accuracy | macro-F1 | AUC | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| core3 supervised error-router MoE | 0.7756 | 0.7318 | 0.8979 | 0.7400 | 0.6092 | 0.8582 |

However, the follow-up reveals useful structure:

- Margin-weighted routing is a strong lever for `Enrolled` recall.
- Meta-router features improve AUC / uncertainty awareness but do not improve macro-F1 here.
- One-step M-step expert specialization is not currently reliable.
- Linear/SVM heterogeneity is real but too weak unless used as a targeted `Enrolled` specialist rather than a full third expert replacement.

The best next direction is not another broad expert replacement. It is likely a constrained class-aware router objective: use the ElasticNet expert only as an `Enrolled` specialist under controlled probability blending, while retaining tree experts for `Dropout` and `Graduate`.

## Direct Weighted Vote Of Two MoE Variants

After the failed two-variant logistic stacking attempt was discarded, a direct probability vote was tested using only two standalone MoE variants:

- `moe_best_macro_raw_logistic_none_soft`
- `moe_enrolled_raw_logistic_sqrt_margin_soft`

Implementation detail:

- The standalone MoE probabilities are cached once in `models/metrics/supervised_moe_variant_probabilities.npz`.
- The direct vote script only reads this cache and does not train experts, routers, or a stacker.
- OOF macro-F1 selects the weighted vote.

Scripts:

- `scripts/ensemble/save_supervised_moe_variant_probabilities.py`
- `scripts/ensemble/weighted_merge_supervised_moe_variants.py`

Outputs:

- `models/metrics/supervised_moe_variant_probabilities.npz`
- `models/metrics/supervised_moe_variant_probability_cache_metrics.csv`
- `models/metrics/weighted_merge_supervised_moe_variants_metrics.csv`
- `models/metrics/weighted_merge_supervised_moe_variants_weight_grid.csv`
- `models/metrics/weighted_merge_supervised_moe_variants_weights.json`

Result:

| model | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best standalone MoE | 0.7756 | 0.7318 | 0.8979 | 0.6867 | 0.7400 | 0.6092 | 0.8582 |
| equal soft vote | 0.7733 | 0.7307 | 0.8980 | 0.6836 | 0.7447 | 0.6176 | 0.8477 |
| OOF-weighted soft vote | 0.7733 | 0.7303 | 0.8981 | 0.6838 | 0.7447 | 0.6134 | 0.8492 |
| enrolled-oriented MoE | 0.7703 | 0.7290 | 0.8970 | 0.6798 | 0.7400 | 0.6303 | 0.8401 |

OOF-selected weights:

| variant | weight |
| --- | ---: |
| best standalone MoE | 0.62 |
| enrolled-oriented MoE | 0.38 |

Finding:

- Direct weighted vote does not improve macro-F1 over the best standalone MoE.
- It slightly improves AUC and `Enrolled` recall, but loses too much `Graduate` recall.
- The best claim remains the standalone core3 supervised error-router MoE.

## Switch-To-Enrolled Rule

A more class-aware hard switch was also tested:

```text
if enrolled-oriented MoE predicts Enrolled:
    use enrolled-oriented MoE probabilities
else:
    use best standalone MoE probabilities
```

Script:

- `scripts/ensemble/switch_supervised_moe_on_enrolled.py`

Outputs:

- `models/metrics/switch_supervised_moe_on_enrolled_metrics.csv`
- `models/metrics/switch_supervised_moe_on_enrolled_best.json`
- `models/metrics/switch_supervised_moe_on_enrolled_rule.json`

Result:

| model | accuracy | macro-F1 | AUC | FMI | dropout acc | enrolled acc | graduate acc | switch count | switch share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best standalone MoE | 0.7756 | 0.7318 | 0.8979 | 0.6867 | 0.7400 | 0.6092 | 0.8582 | n/a | n/a |
| switch-to-Enrolled rule | 0.7688 | 0.7295 | 0.8976 | 0.6771 | 0.7307 | 0.6513 | 0.8356 | 310 | 0.2334 |
| enrolled-oriented MoE | 0.7703 | 0.7290 | 0.8970 | 0.6798 | 0.7400 | 0.6303 | 0.8401 | n/a | n/a |

Finding:

- The rule works as intended for `Enrolled`: recall rises from 0.6092 to 0.6513.
- It still does not improve macro-F1 because the switched subset contains too many samples that the best MoE handled better for `Dropout` or `Graduate`.
- A useful next refinement would require a confidence gate, not just `argmax == Enrolled`.
