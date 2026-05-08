# FT-Transformer Tuning Ranked

| Rank | Trial | d_token | blocks | attn_do | ff_do | lr | Macro-F1 | AUC | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ft_token80_lower_dropout | 80 | 2 | 0.10 | 0.10 | 0.0005 | 0.7193 | 0.8846 | 0.7545 |
| 2 | ft_token64_lower_dropout | 64 | 2 | 0.10 | 0.10 | 0.0008 | 0.7187 | 0.8956 | 0.7538 |
| 3 | ft_token64_small_lr | 64 | 2 | 0.15 | 0.15 | 0.0006 | 0.7065 | 0.8864 | 0.7410 |
| 4 | ft_token64 | 64 | 2 | 0.20 | 0.20 | 0.0008 | 0.7025 | 0.8861 | 0.7349 |
| 5 | ft_wider_tokens | 48 | 2 | 0.20 | 0.20 | 0.0008 | 0.6941 | 0.8863 | 0.7146 |
| 6 | ft_token80_small_lr | 80 | 2 | 0.15 | 0.15 | 0.0005 | 0.6890 | 0.8815 | 0.7123 |