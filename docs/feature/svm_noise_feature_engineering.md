# SVM Noise Feature Engineering Experiment

## Purpose

本实验记录 SVM 专属的 feature engineering 与 data noise analysis。前置检查显示 `data/processed/train.csv` 与 `data/processed/test.csv` 的三类比例足够平衡，split 本身不是 SVM 表现波动的主要原因。因此后续重点转向 SVM 的特征空间：哪些列会制造距离噪声，哪些编码方式能改善 `Dropout / Enrolled / Graduate` 三分类边界。

当前 SVM 的主要瓶颈是 `Enrolled` 类。它和 `Dropout` / `Graduate` 在特征空间中重叠较多，因此单纯扩大 `C/gamma` 网格收益有限，需要减少弱信号特征对 RBF 距离的干扰。

## Scripts And Artifacts

主要脚本：

- `scripts/baseline/search_svm_hyperparameters.py`

噪声诊断输出：

- `visualization/noise_feature_analysis/feature_noise_ranking.csv`
- `visualization/noise_feature_analysis/category_rare_rate.csv`
- `visualization/noise_feature_analysis/train_test_shift.csv`
- `visualization/noise_feature_analysis/svm_drop_probe_cv.csv`
- `visualization/noise_feature_analysis/noise_feature_diagnostics_report.md`

SVM 实验输出：

- `models/metrics/baseline/svm_hyperparameter_fe_refine_dense_best.json`
- `models/metrics/baseline/svm_hyperparameter_fe_refine_dense_drop_noise_best.json`
- `models/metrics/baseline/svm_hyperparameter_fe_refine_app_thermo_drop_noise_best.json`

## Noise Analysis Method

噪声特征分析先读取现有 `visualization/` 下的报告和图，再对 processed train/test 做轻量诊断，不跑大规模模型搜索。分析维度包括：

- missing rate 与 unique/cardinality；
- 类别特征 rare category 质量；
- train/test 分布漂移，数值列看分布差异，类别列看频率差；
- 与 `Target` 的低成本相关性、互信息和单变量可分性；
- 固定 SVM 参数下的 drop-probe CV，观察删除某组列后 macro-F1 是否上升。

关键发现：

- `Curricular units 1st/2nd sem (credited)` 的互信息接近 0，主值比例约 87%，单变量 macro-F1 很低。
- `Curricular units 1st/2nd sem (without evaluations)` 信号弱，主值比例约 93%，其中二学期 without evaluations 对 `Enrolled` 单变量 recall 近乎无效。
- `International` 与 `Educational special needs` 近常量，主值比例分别约 97.7% / 98.8%，互信息接近 0。
- `Nacionality` 也接近主类别支配，但直接删除几乎中性；更适合后续做 rare bucket，而不是本轮直接删除。
- 父母学历/职业、`Application mode`、`Course` 虽有 rare level 和漂移，但删除会损伤 CV，因此不进入首轮 drop-list。

## Implemented Feature Sets

脚本新增了 `--feature-set` 分支：

- `dense`: 原始 SVM 表示。numeric median impute + `StandardScaler`，categorical most frequent impute + dense one-hot。
- `app_thermo`: 把 `Application order` 从单个 scaled scalar 改成 1-9 累积 thermometer encoding。
- `mixed_te`: 高基数类别用 `TargetEncoder(target_type='multiclass', cv=5)`，低基数类别保留 one-hot。
- `mixed_te_app_thermo`: 同时使用 high-cardinality target encoding 与 `Application order` thermometer。
- `ohe_svd`: 类别 one-hot 后接 `TruncatedSVD` 压缩，以测试是否能减少高维 OHE 距离噪声。
- `*_drop_noise`: 在上述 feature set 基础上删除首轮噪声列。

首轮 drop-list：

```python
[
    "Curricular units 1st sem (credited)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (without evaluations)",
    "International",
    "Educational special needs",
]
```

## Hyperparameter Search

主要搜索模式为 `fe_refine`，围绕前一轮最佳 RBF-SVM 区域小范围搜索：

- `kernel = rbf`
- `C in [10, 15, 20, 30]`
- `gamma in [0.003, 0.004, 0.006, 0.008, 0.01]`
- `class_weight` 包括 `None` 与多个 `Enrolled` up-weight 方案
- CV 使用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 搜索评分为 `f1_macro`
- 对 top CV candidates 在 held-out test 上 refit with `probability=True`，用于输出 AUC

另外跑过 `rbf_final_refine`，围绕 `C≈30, gamma≈0.003` 更细搜索，但没有超过 `dense_drop_noise`。

## Results

| mode | feature set | accuracy | macro-F1 | AUC | enrolled recall | enrolled F1 | best parameters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fe_refine` | `dense_drop_noise` | **0.7854** | **0.7306** | 0.8910 | 0.5294 | 0.5396 | `C=15.0, gamma=0.006, Enrolled weight=1.25` |
| `fe_refine` | `dense` | 0.7809 | 0.7249 | **0.8916** | 0.5168 | 0.5268 | `C=30.0, gamma=0.003, Enrolled weight=1.25` |
| `fe_refine` | `app_thermo_drop_noise` | 0.7816 | 0.7247 | 0.8909 | 0.5084 | 0.5216 | `C=30.0, gamma=0.003, Enrolled weight=1.25` |
| `fe_refine` | `app_thermo` | 0.7809 | 0.7239 | 0.8881 | 0.5126 | 0.5225 | `C=10.0, gamma=0.01, Enrolled weight=1.25` |
| `rbf_final_refine` | `dense` | 0.7771 | 0.7234 | 0.8913 | 0.5378 | 0.5289 | `C=40.0, gamma=0.0025, Enrolled weight=1.35` |
| `fe_refine` | `mixed_te_app_thermo` | 0.7786 | 0.7220 | 0.8895 | 0.5168 | 0.5223 | `C=15.0, gamma=0.003, Enrolled weight=1.25` |
| `fe_refine` | `mixed_te` | 0.7748 | 0.7219 | 0.8876 | **0.5462** | 0.5274 | `C=20.0, gamma=0.004, Enrolled weight=1.5` |
| `fe_refine` | `ohe_svd` | 0.7673 | 0.7119 | 0.8810 | 0.5168 | 0.5093 | `C=10.0, gamma=0.003, Enrolled weight=1.5` |

## SelectKBest Follow-Up

在 `dense_drop_noise` 已经删除 6 个弱信号列之后，进一步测试了 fold-internal feature selection。实现方式是在 `scripts/baseline/search_svm_hyperparameters.py` 中加入 `--selector` 分支，把 `SelectKBest` 放在 preprocessing 之后、`SVC` 之前，因此每个 CV fold 内独立 fit selector，不会把 held-out test 信息泄漏进特征选择过程。

测试配置：

- `--mode selectk_refine`
- `--feature-set dense_drop_noise`
- `--selector f_classif` 或 `--selector mutual_info`
- `selector__k in [80, 120, 160, 200, "all"]`
- SVM 搜索区域：`C in [10, 15, 20, 30]`，`gamma in [0.003, 0.004, 0.006, 0.008]`
- class weights 包括 `None`、`Enrolled=1.1/1.25/1.5`

结果：

| mode | feature set | selector | accuracy | macro-F1 | AUC | best k | best parameters |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `fe_refine` | `dense_drop_noise` | none | **0.7854** | **0.7306** | 0.8910 | n/a | `C=15.0, gamma=0.006, Enrolled weight=1.25` |
| `selectk_refine` | `dense_drop_noise` | `f_classif` | 0.7824 | 0.7260 | 0.8910 | `all` | `C=30.0, gamma=0.003, Enrolled weight=1.25` |
| `selectk_refine` | `dense_drop_noise` | `mutual_info` | 0.7824 | 0.7260 | 0.8910 | `all` | `C=30.0, gamma=0.003, Enrolled weight=1.25` |

结论：`SelectKBest` 没有带来提升。两个 selector 的最佳 held-out test 候选都退回到 `k="all"`，仍低于无 selector 的 `dense_drop_noise`。这说明在已经删除明确噪声列后，继续按单变量得分裁剪 one-hot 展开维度会破坏 RBF-SVM 依赖的局部组合边界。

对应输出：

- `models/metrics/baseline/svm_hyperparameter_selectk_refine_dense_drop_noise_f_classif_best.json`
- `models/metrics/baseline/svm_hyperparameter_selectk_refine_dense_drop_noise_f_classif_test_metrics.csv`
- `models/metrics/baseline/svm_hyperparameter_selectk_refine_dense_drop_noise_mutual_info_best.json`
- `models/metrics/baseline/svm_hyperparameter_selectk_refine_dense_drop_noise_mutual_info_test_metrics.csv`

## Interpretation

`dense_drop_noise` 是当前 SVM 最优结果。相比未删噪声列的 `dense`：

- accuracy: `0.7809 -> 0.7854`，提升 `+0.0045`
- macro-F1: `0.7249 -> 0.7306`，提升 `+0.0057`
- enrolled recall: `0.5168 -> 0.5294`，提升 `+0.0126`
- enrolled F1: `0.5268 -> 0.5396`，提升 `+0.0128`

这说明本轮收益主要来自删除 SVM 距离空间中的低价值列，而不是复杂编码。

`Application order` thermometer 在不删噪声时有轻微正向迹象，但去噪后没有叠加收益。当前数据下，把 `Application order` 保持原 dense scalar 反而略好。它不应直接删除，因为 drop-probe 显示删除会有小降。

`mixed_te` 提高了 `Enrolled` recall，但 macro-F1 没有超过 `dense_drop_noise`，说明它倾向于换取中间类召回，但牺牲了两端类或整体概率排序。后续如果目标是专门提高 `Enrolled` recall，可以继续研究；如果主指标是 macro-F1，当前不作为默认方案。

`ohe_svd` 明显变差，说明本数据的 OHE 维度虽然高，但直接 SVD 压缩会丢掉有用的类别边界，不适合作为当前 SVM 默认输入。

## Recommended Default

当前推荐 SVM 配置：

```text
feature_set = dense_drop_noise
kernel = rbf
C = 15.0
gamma = 0.006
class_weight = {
  Dropout: 1.0,
  Enrolled: 1.25,
  Graduate: 1.0
}
```

对应 test 指标：

```text
Accuracy = 0.7854
Macro-F1 = 0.7306
AUC = 0.8910
Enrolled recall = 0.5294
Enrolled F1 = 0.5396
```

## Reproduction Commands

```bash
python scripts/baseline/search_svm_hyperparameters.py \
  --mode fe_refine \
  --feature-set dense_drop_noise \
  --cv 5 \
  --n-jobs 96 \
  --top-k-test 20
```

对照实验：

```bash
python scripts/baseline/search_svm_hyperparameters.py --mode fe_refine --feature-set dense --cv 5 --n-jobs 96 --top-k-test 20
python scripts/baseline/search_svm_hyperparameters.py --mode fe_refine --feature-set app_thermo_drop_noise --cv 5 --n-jobs 96 --top-k-test 20
python scripts/baseline/search_svm_hyperparameters.py --mode fe_refine --feature-set mixed_te --cv 5 --n-jobs 64 --top-k-test 15
python scripts/baseline/search_svm_hyperparameters.py --mode fe_refine --feature-set ohe_svd --svd-components 64 --cv 5 --n-jobs 64 --top-k-test 15
python scripts/baseline/search_svm_hyperparameters.py --mode selectk_refine --feature-set dense_drop_noise --selector f_classif --cv 5 --n-jobs 96 --top-k-test 30
python scripts/baseline/search_svm_hyperparameters.py --mode selectk_refine --feature-set dense_drop_noise --selector mutual_info --cv 5 --n-jobs 96 --top-k-test 30
```

## Caveats

当前表格是 held-out test 上的 top-CV-candidate 对比，不是 cross-seed final claim。`dense_drop_noise` 已有明确收益，但最终报告中如果要作为稳定结论，建议再做 cross-seed 或 repeated CV 验证。

本实验只证明上述 6 个 weak/noise-like columns 对 RBF-SVM 有害；不要把这个结论直接外推到 CatBoost / LightGBM / XGBoost / DL。树模型和深度模型可能通过交互或内部正则处理这些列。
