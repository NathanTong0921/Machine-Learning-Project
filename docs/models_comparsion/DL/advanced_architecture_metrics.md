# Advanced Architecture Metrics

## Scope

This note summarizes the held-out test performance of the advanced/open-ended architectures tried in this project. The primary comparison metric is `f1_macro`, because the `Enrolled` class is smaller and harder than `Dropout` and `Graduate`; accuracy and macro AUC are reported as secondary metrics.

## Raw Comparison Table

| Family | Model | Source | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Macro AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | `multiview_gated_mlp_best` | `advanced_model_comparison.csv` | 0.7575 | 0.7306 | 0.7340 | 0.7217 | 0.7683 | 0.8925 |
| MLP | `tabm_style_multiexpert_mlp` | `tabm_style_comparison.csv` | 0.7636 | 0.7180 | 0.7244 | 0.7186 | 0.7700 | 0.8888 |
| MLP | `multiview_cost_sensitive_enrolled_x1_15` | `multiview_cost_sensitive_comparison.csv` | 0.7530 | 0.7315 | 0.7365 | 0.7201 | 0.7661 | 0.8920 |
| MLP | `multiview_cost_sensitive_enrolled_x1_30_dropout_015` | `multiview_cost_sensitive_comparison.csv` | 0.7485 | 0.7363 | 0.7376 | 0.7179 | 0.7632 | 0.8925 |
| Transformer | `ft_transformer_tuned` | `structured_transformer_comparison.csv` | 0.7515 | 0.7162 | 0.7080 | 0.7053 | 0.7583 | 0.8771 |
| Transformer | `viewwise_gated_ft_transformer` | `viewwise_gated_transformer_comparison.csv` | 0.7417 | 0.7170 | 0.7209 | 0.7066 | 0.7548 | 0.8875 |
| Transformer | `attention_viewwise_gated_ft_transformer` | `attention_viewwise_transformer_comparison.csv` | 0.7440 | 0.7187 | 0.7199 | 0.7074 | 0.7578 | 0.8818 |
| Transformer | `trompt_style_transformer` | `trompt_style_comparison.csv` | 0.7312 | 0.7177 | 0.7056 | 0.6940 | 0.7468 | 0.8779 |
| Transformer | `piecewise_ft_transformer` | `piecewise_numerical_comparison.csv` | 0.6965 | 0.6871 | 0.6811 | 0.6655 | 0.7189 | 0.8666 |
| Foundation tabular | `tabpfn` | `tabpfn_metrics.csv` | 0.7861 | 0.7351 | 0.7030 | 0.7131 | 0.7752 | 0.9017 |

## Key Findings

1. The best advanced neural architecture by macro F1 is `multiview_gated_mlp_best` with macro F1 `0.7217`, accuracy `0.7575`, and macro AUC `0.8925`. It is more balanced than TabPFN on class recall, which matters for the minority `Enrolled` class.

2. TabPFN has the best accuracy and AUC among the advanced architectures: accuracy `0.7861` and macro AUC `0.9017`. However, its macro recall is only `0.7030`, so its macro F1 `0.7131` is lower than the best MLP. This suggests strong overall ranking/probability quality but weaker balanced class recovery.

3. Cost-sensitive MLP variants increased `Enrolled` recall but did not improve overall macro F1. For example, `multiview_cost_sensitive_enrolled_x1_15` reached macro recall `0.7365`, but macro F1 stayed at `0.7201`, slightly below the best MLP baseline.

4. TabM-style multi-expert MLP improved accuracy to `0.7636`, but macro F1 `0.7186` remained below `multiview_gated_mlp_best`. It is a useful open-ended comparison, but not the best final model.

5. FT-Transformer variants did not outperform the MLP family on this dataset. The best transformer-style variants are attention/viewwise gated FT models around macro F1 `0.7066-0.7074`, below the best MLP by about `0.014-0.015`.

6. Piecewise numerical tokenization was harmful in this run: `piecewise_ft_transformer` dropped to macro F1 `0.6655`, indicating that this dataset likely does not benefit from that tokenization setting without more tuning.

## Report-Ready Interpretation

The open-ended exploration shows that the strongest deep architecture is the Multi-View Gated MLP. Its view split matches the dataset structure: enrollment background, demographic/family information, first-semester performance, and second-semester performance. TabPFN is highly competitive in accuracy and AUC, but it is less balanced under macro-F1, which is more appropriate for this imbalanced three-class task. Transformer variants are interesting but did not outperform the simpler MLP family, suggesting that the dataset size and feature structure favor targeted inductive bias over heavier sequence-style tabular modeling.

## Suggested Next Experiments

1. Keep `multiview_gated_mlp_best` as the representative neural architecture in the report.
2. Present TabPFN as a strong external/foundation-model baseline, emphasizing its high accuracy and AUC but weaker macro-F1.
3. Use the OOF probability ensemble as the final predictive model if the report focuses on best overall performance, and use this note to explain the advanced neural model branch.
4. If more time is available, tune the MLP class-weight schedule around `Enrolled` using validation macro-F1 instead of only changing a single class multiplier.
