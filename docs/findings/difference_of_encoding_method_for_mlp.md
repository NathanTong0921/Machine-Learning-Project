# Difference Of Encoding Method For MLP

## Question

`Application order` is an ordinal feature: it records the student's application preference order.

The current Appendix-aligned preprocessing keeps it as a numeric/ordinal scalar. For MLP-style models,
this means the feature is median-imputed and standardized, which implicitly treats the gaps between
application orders as approximately continuous and equally spaced.

This is acceptable for a clean baseline, but it can be a scale-bias issue for linear, distance-based,
or neural models. Therefore, we tested whether ordinal-aware encodings improve the MultiView Gated MLP.

## Compared Encodings

| encoding | description | enrollment view dim |
| --- | --- | ---: |
| scalar | Keep `Application order` as one standardized numeric scalar | 80 |
| thermometer | Replace scalar with 9 cumulative indicators: `Application order >= 1..9` | 88 |
| OOF target encoding | Replace scalar with 3 leakage-controlled smoothed class-rate features | 82 |

Target encoding was generated with 5-fold OOF features for the training set. The test set used full-train
smoothed mappings.

## Fixed-Hyperparameter Result

First, we used the scalar-tuned MLP configuration and only changed the encoding. This made the alternatives
look worse:

| encoding | accuracy | macro-F1 | AUC | Dropout acc | Enrolled acc | Graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar | 0.7718 | 0.7285 | 0.8889 | 0.7963 | 0.5882 | 0.8220 |
| thermometer | 0.7553 | 0.7153 | 0.8888 | 0.6838 | 0.6471 | 0.8401 |
| OOF target encoding | 0.7568 | 0.7158 | 0.8902 | 0.7166 | 0.6261 | 0.8296 |

This result was misleading because changing the encoding also changes input dimension and optimization behavior.

## Retuned Result

After retuning the MLP for each encoding, the conclusion changed:

| encoding | accuracy | macro-F1 | weighted-F1 | FMI | AUC | Dropout acc | Enrolled acc | Graduate acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar tuned | 0.7718 | 0.7285 | 0.7777 | 0.6808 | 0.8889 | 0.7963 | 0.5882 | 0.8220 |
| thermometer tuned | 0.7824 | 0.7350 | 0.7840 | 0.6970 | 0.8903 | 0.7447 | 0.5798 | 0.8793 |
| OOF target encoding tuned | 0.7681 | 0.7299 | 0.7778 | 0.6787 | 0.8949 | 0.6956 | 0.6765 | 0.8477 |

Best thermometer configuration:

```json
{
  "seed": 62,
  "hidden_dim": 128,
  "dropout": 0.05,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 64,
  "max_epochs": 90,
  "patience": 12,
  "class_weight_multipliers": {}
}
```

Best OOF target-encoding configuration:

```json
{
  "seed": 62,
  "hidden_dim": 160,
  "dropout": 0.20,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 64,
  "max_epochs": 90,
  "patience": 12,
  "class_weight_multipliers": {}
}
```

## Interpretation

The fixed-hyperparameter result underestimates the alternative encodings. Once retuned, thermometer encoding
becomes the best MultiView MLP treatment for `Application order`.

Main observations:

- Thermometer encoding improves macro-F1 from `0.7285` to `0.7350`.
- Thermometer encoding also improves accuracy from `0.7718` to `0.7824`.
- Its main gain is stronger `Graduate` recall, rising from `0.8220` to `0.8793`.
- OOF target encoding gives the best AUC (`0.8949`) and best `Enrolled` recall (`0.6765`), but lower macro-F1 than thermometer.
- Scalar encoding has the strongest `Dropout` recall (`0.7963`), but is weaker on overall macro-F1.

## Conclusion

For the final MLP-side setting, use:

```text
MultiView Gated MLP + thermometer Application order encoding
```

This result also supports the review concern: treating `Application order` as a standardized scalar is a reasonable
baseline, but it is not necessarily optimal for MLP. Ordinal-aware encoding should be used when optimizing the MLP branch.

## Artifacts

- `scripts/mlp/compare_application_order_encodings_multiview.py`
- `scripts/mlp/search_application_order_encoding_multiview.py`
- `models/metrics/multiview_application_order_encoding_metrics.csv`
- `models/metrics/multiview_application_order_encoding_search_thermometer.csv`
- `models/metrics/multiview_application_order_encoding_search_target_encoding_oof.csv`
- `models/metrics/multiview_application_order_encoding_tuned_summary.csv`
- `docs/compare_application_order_encodings_multiview.md`
