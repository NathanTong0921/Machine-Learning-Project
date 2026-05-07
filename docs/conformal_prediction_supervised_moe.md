# Conformal Prediction On Core3 MoE

## Purpose

This experiment changes the output form instead of changing the base model.

The base model remains the current best pure-ML Core3 supervised error-router MoE. Instead of forcing every row into one hard class, conformal prediction outputs a prediction set:

- Clear case: `{Dropout}`
- Ambiguous case: `{Enrolled,Graduate}`
- Highly uncertain case: `{Dropout,Enrolled,Graduate}`

This is useful because `Enrolled` and `Graduate` overlap strongly in feature space. Forcing a hard label can create errors where a set-valued answer is more honest.

## Script And Outputs

Script:

- `scripts/ensemble/conformal_predict_supervised_moe.py`

Inputs:

- `models/metrics/supervised_moe_variant_probabilities.npz`

Outputs:

- `models/metrics/conformal_supervised_moe_best_metrics.csv`
- `models/metrics/conformal_supervised_moe_best_test_sets.csv`
- `models/metrics/conformal_supervised_moe_best_config.json`

Command:

```bash
python scripts/ensemble/conformal_predict_supervised_moe.py --variant best --alphas 0.05,0.10,0.15,0.20
```

## Calibration Note

This implementation uses OOF train probabilities as calibration scores. This is a cross-conformal-style diagnostic and is leakage-aware in the sense that each train row's probability comes from a model that did not train on that row.

For a strict split-conformal guarantee, reserve a standalone calibration split and do not tune on it.

## Methods

THR:

- Nonconformity score: `1 - p_true`
- Prediction set contains labels whose predicted probability is high enough.
- Tends to produce smaller sets.

APS:

- Nonconformity score: cumulative sorted probability up to the true label.
- Prediction set adapts to probability shape.
- Tends to include more labels for uncertain minority cases.

## Results

All rows below are evaluated on the held-out test split.

| method | alpha | nominal coverage | observed coverage | avg set size | singleton rate | `{Enrolled,Graduate}` rate | Dropout coverage | Enrolled coverage | Graduate coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| THR | 0.05 | 0.95 | 0.9571 | 1.7696 | 0.4111 | 0.2221 | 0.9227 | 0.9286 | 0.9894 |
| THR | 0.10 | 0.90 | 0.9089 | 1.4322 | 0.6032 | 0.1988 | 0.8571 | 0.8613 | 0.9593 |
| THR | 0.15 | 0.85 | 0.8547 | 1.2380 | 0.7666 | 0.1227 | 0.8126 | 0.7647 | 0.9140 |
| THR | 0.20 | 0.80 | 0.7989 | 1.0753 | 0.9051 | 0.0505 | 0.7541 | 0.6555 | 0.8793 |
| APS | 0.05 | 0.95 | 0.9925 | 2.8313 | 0.0000 | 0.0873 | 0.9859 | 0.9958 | 0.9955 |
| APS | 0.10 | 0.90 | 0.9036 | 1.8185 | 0.1182 | 0.4887 | 0.7775 | 0.9454 | 0.9698 |
| APS | 0.15 | 0.85 | 0.8667 | 1.6461 | 0.2410 | 0.4164 | 0.7026 | 0.8992 | 0.9608 |
| APS | 0.20 | 0.80 | 0.8140 | 1.4827 | 0.3306 | 0.3502 | 0.5972 | 0.8613 | 0.9367 |

## Interpretation

THR is the more compact option:

- At nominal 90% coverage, observed coverage is 0.9089.
- Average set size is 1.4322.
- 60.3% of test rows still receive a singleton label.

APS is better for ambiguity-aware reporting:

- At nominal 90% coverage, observed coverage is 0.9036.
- Enrolled coverage is 0.9454, much higher than THR's 0.8613.
- It produces `{Enrolled,Graduate}` for 48.9% of test rows, which directly reflects the overlap between these two classes.

## Finding

Conformal prediction does not improve hard-label macro-F1 because it changes the task. It gives a scientifically cleaner output for ambiguous student states.

Recommended reporting:

- Use hard-label macro-F1 for the course metric.
- Add conformal prediction as an uncertainty-aware analysis.
- Highlight THR at alpha 0.10 as the compact set-valued predictor.
- Highlight APS at alpha 0.10 as the Enrolled/Graduate ambiguity detector.

This is especially useful for explaining why `Enrolled` is structurally hard: the model often needs a valid prediction set rather than a forced single label.
