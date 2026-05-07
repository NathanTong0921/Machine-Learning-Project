# search_multiview_current_preprocess.py

对应脚本：`scripts/mlp/search_multiview_current_preprocess.py`

## 目的

旧的 `multiview_gated_mlp_best.pt` 是在 legacy preprocessing 下训练的：`Application order`
被当作 categorical one-hot，因此 `enrollment_profile` 输入维度为 `87`。

当前 Appendix-aligned preprocessing 把 `Application order` 作为 numeric/ordinal，因此
`enrollment_profile` 输入维度变为 `80`。直接沿用旧 best 超参重训后，macro-F1 从 `0.7217`
降到 `0.7089`，所以重新搜索 MultiView Gated MLP 超参。

## 运行方式

两张 HIP/ROCm GPU 分片并行：

```bash
env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  python scripts/mlp/search_multiview_current_preprocess.py --shard-index 0 --num-shards 2

env HIP_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=0 \
  python scripts/mlp/search_multiview_current_preprocess.py --shard-index 1 --num-shards 2
```

搜索空间共 `46` 组配置，主要覆盖：

- `hidden_dim`: 64, 96, 128, 160
- `dropout`: 0.05, 0.10, 0.15, 0.20
- `lr`: 5e-4, 7e-4, 1e-3, 1.5e-3
- `batch_size`: 64, 128, 256
- `class_weight_multipliers`: baseline, `{"Enrolled": 1.15}`, `{"Enrolled": 1.30}`

## 结果

| model | preprocessing | accuracy | macro-F1 | weighted-F1 | FMI | AUC | Dropout acc | Enrolled acc | Graduate acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old `multiview_gated_mlp_best` | legacy: `Application order` categorical | 0.7575 | 0.7217 | 0.7683 | 0.6611 | 0.8925 | 0.6979 | 0.6807 | 0.8235 |
| current preprocess, old hyperparams | current: `Application order` numeric | 0.7492 | 0.7089 | 0.7586 | 0.6535 | 0.8886 | 0.7213 | 0.6176 | 0.8145 |
| current preprocess, searched best | current: `Application order` numeric | 0.7718 | 0.7285 | 0.7777 | 0.6808 | 0.8889 | 0.7963 | 0.5882 | 0.8220 |

最佳配置：

```json
{
  "seed": 62,
  "hidden_dim": 96,
  "dropout": 0.1,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "batch_size": 64,
  "max_epochs": 90,
  "patience": 12,
  "class_weight_multipliers": {}
}
```

## 差距

相对旧 legacy best：

| compared model | accuracy delta | macro-F1 delta | weighted-F1 delta | FMI delta | AUC delta | Dropout acc delta | Enrolled acc delta | Graduate acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current preprocess, old hyperparams | -0.0083 | -0.0129 | -0.0096 | -0.0075 | -0.0039 | +0.0234 | -0.0630 | -0.0090 |
| current preprocess, searched best | +0.0143 | +0.0068 | +0.0095 | +0.0198 | -0.0036 | +0.0984 | -0.0924 | -0.0015 |

相对当前 preprocess 的旧超参重训：

| compared model | accuracy delta | macro-F1 delta | weighted-F1 delta | FMI delta | AUC delta | Dropout acc delta | Enrolled acc delta | Graduate acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| searched best | +0.0226 | +0.0196 | +0.0191 | +0.0273 | +0.0003 | +0.0749 | -0.0294 | +0.0075 |

## 结论

当前 Appendix-aligned preprocessing 不是失败点。旧超参在输入空间从 `87` 改为 `80` 后不再合适；
重新搜索后，当前 preprocess 的 MultiView Gated MLP 可以达到 `0.7285` macro-F1，超过旧 legacy
checkpoint 的 `0.7217`。

代价是类别分布偏向两端：Dropout acc 明显提高，但 Enrolled acc 从旧 best 的 `0.6807` 降到 `0.5882`。
如果报告重点是 macro-F1，可以采用 searched best；如果重点是 Enrolled recall，旧 cost-sensitive MLP
或 two-stage threshold tuning 仍然值得作为补充分析。

## 输出文件

- `models/metrics/multiview_current_preprocess_search_all.csv`
- `models/metrics/multiview_current_preprocess_search_best.json`
- `models/metrics/multiview_current_preprocess_search_shard0_of2.csv`
- `models/metrics/multiview_current_preprocess_search_shard1_of2.csv`
- `models/saved_models/multiview_current_preprocess_search_best.pt`
- `models/saved_models/multiview_current_preprocess_search_best_shard0_of2.pt`
- `models/saved_models/multiview_current_preprocess_search_best_shard1_of2.pt`
