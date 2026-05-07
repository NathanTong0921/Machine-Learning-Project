# Best Model Hyperparameters

## Short Answer

Yes, the main hyperparameters for the best-performing results are saved, but they are split across several files:

- Final ensemble weights are saved in JSON.
- Tuned CatBoost and LightGBM hyperparameters are saved in JSON.
- Neural architecture metadata is saved, but some explicit training configs, such as `hidden_dim`, `dropout`, and `lr`, are still most clearly recorded in the corresponding scripts.

## Best Held-Out Test Results

| Selection rule | Model | Test accuracy | Test macro F1 | Test macro AUC | Saved configuration |
| --- | --- | ---: | ---: | ---: | --- |
| Best macro F1 among final ensemble variants | `oof_probability_ensemble_semester_members_test` | 0.7824 | 0.7343 | 0.9013 | `models/metrics/oof_probability_ensemble_semester_members_best_weights.json` |
| Best accuracy / macro AUC among final ensemble variants | `oof_probability_ensemble_test` | 0.7869 | 0.7320 | 0.9041 | `models/metrics/oof_probability_ensemble_best_weights.json` |

## Final Semester-Member Ensemble

Source metrics:

- `models/metrics/oof_probability_ensemble_semester_members_metrics.csv`

Saved best weights:

- `models/metrics/oof_probability_ensemble_semester_members_best_weights.json`

Members:

| Member | Weight |
| --- | ---: |
| `catboost_tuned` | 0.00 |
| `lightgbm_tuned` | 0.30 |
| `multiview_gated_mlp_best` | 0.00 |
| `tabpfn` | 0.20 |
| `catboost_tuned_semester` | 0.50 |
| `lightgbm_tuned_semester` | 0.00 |

Selection details:

- Objective: maximize OOF macro-F1.
- Coarse grid step: `0.05`.
- Refined active-member step: `0.025`.
- Best OOF macro-F1: `0.7389`.
- Held-out test set used once for final evaluation.

## Base OOF Ensemble

Source metrics:

- `models/metrics/oof_probability_ensemble_metrics.csv`

Saved best weights:

- `models/metrics/oof_probability_ensemble_best_weights.json`

Members:

| Member | Weight |
| --- | ---: |
| `catboost_tuned` | 0.50 |
| `lightgbm_tuned` | 0.00 |
| `multiview_gated_mlp_best` | 0.05 |
| `tabpfn` | 0.45 |

Selection details:

- Objective: maximize OOF macro-F1.
- Weight grid step: `0.05`.
- Best OOF macro-F1: `0.7313`.

## Tuned Tree Member Hyperparameters

### CatBoost

Saved file:

- `models/metrics/catboost_tuned_best_params.json`

Parameters:

```json
{
  "depth": 6,
  "iterations": 500,
  "l2_leaf_reg": 9,
  "learning_rate": 0.05,
  "random_strength": 1.0
}
```

Additional fixed settings in `scripts/boost/tune_catboost_baseline.py` and `scripts/ensemble/ensemble_with_semester_tree_members.py`:

- `loss_function="MultiClass"`
- `eval_metric="TotalF1:average=Macro"`
- `auto_class_weights="Balanced"`
- `verbose=0`

### LightGBM

Saved file:

- `models/metrics/lightgbm_tuned_best_params.json`

Parameters:

```json
{
  "colsample_bytree": 1.0,
  "learning_rate": 0.03,
  "max_depth": -1,
  "min_child_samples": 40,
  "n_estimators": 300,
  "num_leaves": 63,
  "subsample": 0.8
}
```

Additional fixed settings in `scripts/boost/tune_lightgbm_baseline.py` and `scripts/ensemble/ensemble_with_semester_tree_members.py`:

- `objective="multiclass"`
- `class_weight="balanced"`
- `n_jobs=108` for the semester-member ensemble script.

## Advanced Architecture Configs

### Multi-View Gated MLP

Best advanced architecture result:

- Metrics: `models/metrics/advanced_model_comparison.csv`
- Metadata: `models/metrics/multiview_gated_mlp_best_metadata.json`
- Checkpoint: `models/saved_models/multiview_gated_mlp_best.pt`

The explicit best config is currently stored in script constants, not fully repeated in the metadata JSON:

```python
BEST_MV_CONFIG = {
    "seed": 62,
    "hidden_dim": 64,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
}
```

Metadata saved in `multiview_gated_mlp_best_metadata.json` includes:

- `device`
- `seed`
- `best_epoch`
- `best_val_loss`
- `view_dims`
- training/validation loss history

### TabM-Style Multi-Expert MLP

Metrics:

- `models/metrics/tabm_style_comparison.csv`

Metadata:

- `models/metrics/tabm_style_multiexpert_mlp_metadata.json`

Script config in `scripts/mlp/compare_tabm_style_models.py`:

```python
TABM_STYLE_CONFIG = {
    "seed": 62,
    "hidden_dim": 64,
    "dropout": 0.10,
    "num_experts": 8,
    "expert_hidden_dim": 96,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 80,
    "patience": 12,
}
```

### FT-Transformer

Tuning output:

- `models/metrics/ft_tuning_best.json`

Best tuned record:

```json
{
  "trial_name": "ft_wider_tokens",
  "seed": 62,
  "d_token": 48,
  "n_blocks": 2,
  "attention_n_heads": 4,
  "attention_dropout": 0.2,
  "ff_dropout": 0.2,
  "lr": 0.0008,
  "weight_decay": 0.0001,
  "batch_size": 128
}
```

## Reproducibility Gap

The final ensemble and tuned tree parameters are saved clearly. The main gap is that some neural architecture configs are saved in script constants rather than duplicated into their metadata JSON files. For final submission, this document should be cited together with the relevant scripts, or the scripts can be updated later to persist each config dictionary into the metadata JSON.
