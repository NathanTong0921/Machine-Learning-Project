# FT-Transformer Tuning Ranked

| Rank | Trial | d_token | blocks | attn_do | ff_do | lr | Macro-F1 | AUC | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ft_wider_tokens | 48 | 2 | 0.20 | 0.20 | 0.0008 | 0.7033 | 0.8837 | 0.7402 |
| 2 | ft_token64_lower_dropout | 64 | 2 | 0.10 | 0.10 | 0.0008 | 0.6960 | 0.8809 | 0.7282 |
| 3 | ft_token64_small_lr | 64 | 2 | 0.15 | 0.15 | 0.0006 | 0.6891 | 0.8782 | 0.7191 |
| 4 | ft_token80_lower_dropout | 80 | 2 | 0.10 | 0.10 | 0.0005 | 0.6872 | 0.8752 | 0.7123 |
| 5 | ft_token64 | 64 | 2 | 0.20 | 0.20 | 0.0008 | 0.6840 | 0.8804 | 0.7071 |
| 6 | ft_token80_small_lr | 80 | 2 | 0.15 | 0.15 | 0.0005 | 0.6701 | 0.8719 | 0.6898 |