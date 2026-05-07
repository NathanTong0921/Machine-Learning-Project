# DSAA Student Dropout / Enrolled / Graduate 三分类 Valuable Findings

更新时间：2026-05-07

这份笔记只保留当前最值得写进报告/handout 的结论，并明确区分：

- `强证据`：已有成体系数字支撑，可直接写主文。
- `side analysis`：有解释价值，但更适合放分析段或 appendix。
- `needs verification`：当前信号有价值，但还缺 cross-seed / cross-set 复核。

## 1. 预处理与特征编码：`Application order` 的编码方式会实质改变 MLP 结论

### Finding 1.1 `Application order` 对 MLP 不是“无关紧要的小特征”，编码方式会改变最优结果

- 主要数字：
  - scalar tuned：accuracy `0.7718`，macro-F1 `0.7285`，AUC `0.8889`
  - thermometer tuned：accuracy `0.7824`，macro-F1 `0.7350`，AUC `0.8903`
  - OOF target encoding tuned：accuracy `0.7681`，macro-F1 `0.7299`，AUC `0.8949`
  - thermometer 相比 scalar：macro-F1 `+0.0065`，accuracy `+0.0105`
  - OOF target encoding 给出当前最高 Enrolled recall：`0.6765`
- 证据文件路径：
  - `docs/compare_application_order_encodings_multiview.md`
  - `docs/findings/difference_of_encoding_method_for_mlp.md`
  - `models/metrics/multiview_application_order_encoding_tuned_summary.csv`
- 报告可写一句话：
  - 对 MultiView Gated MLP，`Application order` 不是简单保留为 scaled scalar 就足够；重新调参后，thermometer encoding 给出当前最好的 MLP macro-F1，而 OOF target encoding 给出最好的 AUC 和 Enrolled recall。
- 风险/注意事项：
  - 这是单一 train/test split、单主 seed 的结果；编码切换后必须重调超参，否则会低估 thermometer / OOF target encoding。

### Finding 1.2 固定超参会得出错误结论，说明“编码方式”和“超参”强耦合

- 主要数字：
  - 固定 scalar best 超参时，thermometer macro-F1 只有 `0.7153`，OOF target encoding 为 `0.7158`
  - 但重调后分别升到 `0.7350` 和 `0.7299`
- 证据文件路径：
  - `docs/compare_application_order_encodings_multiview.md`
  - `models/metrics/multiview_application_order_encoding_metrics.csv`
  - `models/metrics/multiview_application_order_encoding_tuned_summary.csv`
- 报告可写一句话：
  - `Application order` 编码实验表明，输入表征改变后如果不重调模型，会系统性低估 ordinal-aware encoding 的收益。
- 风险/注意事项：
  - 这条更像方法学注意事项，适合写实验设计说明，不宜夸张成“编码本身一定提升所有模型”。

### Finding 1.3 维度 `80 / 87 / 228 / 241` 有清晰含义，适合在报告里一次讲清楚

- 主要数字：
  - `241`：sklearn one-hot 流水线总输入维度 = `19` numeric + `222` categorical one-hot
  - `228`：当前 MultiView total dim = `80 + 133 + 6 + 9`
  - `80`：当前 `enrollment_profile` view 维度，`Application order` 被当作单个 numeric scalar
  - `87`：旧 legacy `enrollment_profile` 维度，`Application order` 被当作 categorical one-hot
  - 旧 total dim 对应 `235`
  - side experiment 中，thermometer view dim = `88`，OOF target encoding view dim = `82`
- 证据文件路径：
  - `docs/feature_processing_flow.md`
  - `docs/search_multiview_current_preprocess.md`
  - `models/metrics/multiview_application_order_encoding_tuned_summary.csv`
- 报告可写一句话：
  - 本项目的 classical ML 使用 `241` 维 one-hot tabular matrix，而当前最佳 MultiView MLP 使用 `228` 维四视图输入；其中 `Application order` 的编码切换会把 enrollment view 从 `80` 改到 `82` 或 `88` 维。
- 风险/注意事项：
  - `228` 来自 reference seed `62` 的 internal fit 结果；view dim 在别的 split/seed 下可能轻微变化。

### Finding 1.4 不同模型家族吃进去的特征表示并不一样，这本身就是结果差异的一部分

- 主要数字：
  - sklearn baselines / bagging / classic boosting / RF / ExtraTrees：`241` 维 one-hot + scaled numeric
  - CatBoost / LightGBM / XGBoost：原始 `36` 列，使用 native categorical
  - MultiView Gated MLP / TabM-style：视图内 one-hot + scaled numeric
  - FT/Transformer 系列：`x_cat` = `17` 个 categorical ids，`x_num` = `19` 个标准化数值
  - TabPFN：项目侧直接输入原始 `36` 列 DataFrame，无显式 one-hot/scaling
- 证据文件路径：
  - `docs/feature_processing_flow.md`
- 报告可写一句话：
  - 模型比较不是“同一输入上只换 estimator”，而是同时比较了 one-hot、native categorical、embedding-style tokenization 和 raw tabular 输入几种表征范式。
- 风险/注意事项：
  - 报告中应避免把所有模型差异简单归因于架构本身，输入表征也是关键变量。

## 2. One-stage vs Two-stage：阈值校准能救回 Enrolled，但还没超过最强 one-stage

### Finding 2.1 naive two-stage 会明显吞掉 `Enrolled`，threshold tuning 是必要的

- 主要数字：
  - uncalibrated two-stage：macro-F1 `0.7083`，Enrolled recall `0.4748`
  - threshold-tuned two-stage：macro-F1 `0.7217`，Enrolled recall `0.6176`
  - 改变量：macro-F1 `+0.0134`，Enrolled recall `+0.1429`
  - 代价：Dropout recall `-0.0726`，accuracy `-0.0030`
- 证据文件路径：
  - `docs/optimize_two_stage_decision_thresholds.md`
  - `models/metrics/two_stage_threshold_tuned_metrics.csv`
  - `models/metrics/two_stage_vs_one_stage_comparison.csv`
- 报告可写一句话：
  - 对有序 two-stage 分解，真正有价值的不是“直接二阶段化”，而是 OOF threshold tuning；否则中间类 `Enrolled` 会被两端类别吞掉。
- 风险/注意事项：
  - 这个结论目前主要建立在一套 two-stage 成员和一套 held-out split 上，还没有更广的 cross-seed 验证。

### Finding 2.2 threshold-tuned two-stage 没有超过当前最强 one-stage machine ensemble，但它更偏向 `Enrolled`

- 主要数字：
  - machine-only OOF ensemble：accuracy `0.7696`，macro-F1 `0.7247`，AUC `0.8982`，Enrolled recall `0.5924`
  - threshold-tuned two-stage：accuracy `0.7628`，macro-F1 `0.7217`，AUC `0.8925`，Enrolled recall `0.6176`
  - two-stage 相对 one-stage machine ensemble：macro-F1 `-0.0030`，AUC `-0.0057`，Enrolled recall `+0.0252`
- 证据文件路径：
  - `docs/overall_one_stage_two_stage_comparison.md`
  - `docs/optimize_two_stage_decision_thresholds.md`
  - `models/metrics/two_stage_vs_one_stage_comparison.csv`
- 报告可写一句话：
  - 阈值校准后的 two-stage 更适合被解释为一种“提高 Enrolled 召回的 decision strategy”，而不是总体最优模型。
- 风险/注意事项：
  - 若主表只允许放一个最佳传统 ML 结果，应优先放 machine-only one-stage ensemble，而把 two-stage 放进 side analysis。

## 3. Classical ML：最稳主线不是单模型，而是 machine-only ensemble

### Finding 3.1 当前最值得作为传统 ML 主结果的是 machine-only OOF ensemble

- 主要数字：
  - machine-only ensemble test：accuracy `0.7696`，macro-F1 `0.7247`，weighted-F1 `0.7732`，AUC `0.8982`
  - 组成权重：CatBoost tuned `0.30`，LightGBM hist-column best `0.30`，HistGradientBoosting `0.15`，GBDT `0.10`，AdaBoost `0.10`，ExtraTrees `0.05`
- 证据文件路径：
  - `docs/tune_machine_only_ensemble.md`
  - `models/metrics/machine_only_ensemble_metrics.csv`
  - `models/metrics/machine_only_ensemble_best_weights.json`
- 报告可写一句话：
  - 在不引入 MLP / Transformer / TabPFN 的前提下，最佳传统方法不是单个 boosting 模型，而是以 CatBoost 和 LightGBM 为主、再混入 classic boosting 与 ExtraTrees 的 OOF probability ensemble。
- 风险/注意事项：
  - 这是同一 split 上的 OOF 选权结果；虽然比单模型更强，但仍应避免把它描述成“跨数据划分已完全稳定”。

### Finding 3.2 单模型里 CatBoost 仍是最强 classical baseline；LightGBM hist 紧跟，但 tuned LightGBM 没守住优势

- 主要数字：
  - CatBoost tuned：macro-F1 `0.7222`，AUC `0.8959`，Enrolled recall `0.6134`
  - LightGBM hist-column best：macro-F1 `0.7196`，accuracy `0.7718`，AUC `0.8921`
  - baseline CatBoost：macro-F1 `0.7185`
  - tuned LightGBM：macro-F1 `0.7048`，明显低于其 hist-column best
- 证据文件路径：
  - `docs/tune_catboost_baseline.md`
  - `docs/tune_lightgbm_baseline.md`
  - `models/metrics/machine_only_ensemble_metrics.csv`
  - `models/metrics/catboost_tuned_metrics.csv`
  - `models/metrics/lightgbm_tuned_metrics.csv`
  - `models/metrics/lightgbm_metrics.csv`
- 报告可写一句话：
  - 如果只选单个传统表格模型，当前最强的是 tuned CatBoost；LightGBM 的最好结果来自 hist-column baseline，而不是后续 10-fold tuned 版本。
- 风险/注意事项：
  - tuned LightGBM 弱于较早的 hist-column best，说明 LightGBM 这条线对搜索空间和验证协议较敏感，叙述时要避免写成“调参后更强”。

### Finding 3.3 XGBoost accuracy 不差，但对 `Enrolled` 明显失衡，不适合作为主结果

- 主要数字：
  - XGBoost hist-column best：accuracy `0.7696`，macro-F1 `0.6642`
  - Enrolled recall 只有 `0.2479`
  - Graduate recall 高达 `0.9412`
- 证据文件路径：
  - `docs/run_xgboost_baseline.md`
  - `models/metrics/machine_only_ensemble_metrics.csv`
  - `models/metrics/xgboost_metrics.csv`
- 报告可写一句话：
  - XGBoost 在整体 accuracy 上看起来不弱，但它通过强烈偏向两端类别换分，中间类 `Enrolled` 恢复很差，因此不适合用作本任务的主方法。
- 风险/注意事项：
  - 这条结论依赖当前 categorical/native-hist 配置；若以后换损失、类别权重或后处理，行为可能变化。

### Finding 3.4 bagging 家族有补充价值，但没有进入第一梯队

- 主要数字：
  - Pasting Decision Tree：macro-F1 `0.7113`
  - ExtraTrees：macro-F1 `0.7100`
  - Bagging Decision Tree：macro-F1 `0.7079`
  - ExtraTrees 的 Enrolled recall `0.6008`，Bagging 的 Enrolled recall `0.6092`
  - RandomForest：macro-F1 `0.7136`
- 证据文件路径：
  - `docs/search_bagging_tree_models.md`
  - `models/metrics/bagging_tree_metrics.csv`
  - `models/metrics/primary_contenders_summary.csv`
  - `models/metrics/model_comparison_summary.csv`
- 报告可写一句话：
  - tree bagging 方向能给出中等偏强 baseline，但整体仍落后于 tuned CatBoost、LightGBM hist 和 machine-only ensemble。
- 风险/注意事项：
  - bagging 结果可作为 open-ended exploration，但不建议在主文里占太多篇幅。

### Finding 3.5 classic boosting 的主要经验不是“谁绝对赢”，而是不同方法对 `Enrolled` 的 tradeoff 不同

- 主要数字：
  - GBDT log-loss：macro-F1 `0.7112`，Enrolled recall `0.6176`
  - HistGradientBoosting：macro-F1 `0.7096`，Enrolled recall `0.5756`
  - AdaBoost stumps：macro-F1 `0.7021`，Enrolled recall `0.6429`
  - cross-seed / cross-set mean macro-F1：
    - GBDT `0.7237 ± 0.0154`
    - HistGBDT `0.7235 ± 0.0109`
    - AdaBoost `0.7155 ± 0.0151`
- 证据文件路径：
  - `docs/search_classic_boosting_models.md`
  - `docs/verify_classic_boosting_stability.md`
  - `models/metrics/classic_boosting_metrics.csv`
  - `models/metrics/classic_boosting_cross_seed_summary.csv`
- 报告可写一句话：
  - classic boosting 的稳定性分析显示，GBDT 与 HistGradientBoosting 的平均 macro-F1 基本持平；前者略偏 Enrolled，后者更稳，而 AdaBoost 更像 minority-recall 导向对照组。
- 风险/注意事项：
  - 这条是已做 cross-seed/cross-set 验证的少数 classical 发现之一，但它更适合 supporting evidence，不适合压过 CatBoost / ensemble 主线。

### Finding 3.6 cross-seed soft average 值得保留，cross-seed stacking 暂不值得主推

- 主要数字：
  - cross-seed soft average：macro-F1 `0.7247`，AUC `0.8976`，Enrolled recall `0.6092`
  - cross-seed stacking：macro-F1 `0.7172`，AUC `0.8920`，Enrolled recall `0.6765`
  - machine-only OOF ensemble：macro-F1 同样 `0.7247`，但 AUC 更高 `0.8982`
- 证据文件路径：
  - `docs/compare_cross_seed_stacking.md`
  - `models/metrics/cross_seed_stacking_metrics.csv`
  - `models/metrics/machine_only_ensemble_metrics.csv`
- 报告可写一句话：
  - 简单 cross-seed soft averaging 是有效且稳妥的增强，而二层 logistic stacking 虽然显著提高 Enrolled recall，但会牺牲整体 macro-F1。
- 风险/注意事项：
  - 如果报告空间有限，可只保留一句“stacking did not outperform simple averaging”。

## 4. Advanced architectures：最有报告价值的是“MLP 优于 transformer 系列，TabPFN 强在 accuracy/AUC”

### Finding 4.1 当前最佳神经网络主线已经从 scalar MLP 推进到 thermometer MLP

- 主要数字：
  - 旧 best MultiView MLP：macro-F1 `0.7217`
  - current preprocess scalar-tuned MLP：macro-F1 `0.7285`
  - 当前最好 MLP 结果为 thermometer-tuned：accuracy `0.7824`，macro-F1 `0.7350`，AUC `0.8903`
- 证据文件路径：
  - `docs/search_multiview_current_preprocess.md`
  - `docs/compare_application_order_encodings_multiview.md`
  - `models/metrics/multiview_application_order_encoding_tuned_summary.csv`
- 报告可写一句话：
  - 当前最值得代表 MLP 分支的结果不再是旧版 legacy MultiView MLP，而是采用 thermometer `Application order` 编码后重新调参得到的 MultiView Gated MLP。
- 风险/注意事项：
  - 这条是当前最有价值的新发现之一，但仍缺 cross-seed 验证；主文可写成“current best MLP result”而不是“robustly best neural model across seeds”。

### Finding 4.2 MLP 家族整体优于 FT/transformer 系列，说明这份数据更吃结构归纳偏置而不是更重的 token mixer

- 主要数字：
  - FT-Transformer tuned：macro-F1 `0.7053`
  - Viewwise gated FT：`0.7066`
  - Attention viewwise gated FT：`0.7074`
  - Trompt-style：`0.6940`
  - Piecewise FT：`0.6655`
  - 对比当前最好 MLP thermometer：`0.7350`
- 证据文件路径：
  - `docs/advanced_architecture_metrics.md`
  - `models/metrics/structured_transformer_comparison.csv`
  - `models/metrics/viewwise_gated_transformer_comparison.csv`
  - `models/metrics/attention_viewwise_transformer_comparison.csv`
  - `models/metrics/trompt_style_comparison.csv`
  - `models/metrics/piecewise_numerical_comparison.csv`
- 报告可写一句话：
  - 在这份中小规模教育表格数据上，基于视图划分的 gated MLP 明显强于 FT/transformer 系列，说明问题结构更适合 targeted inductive bias，而不是更重的表格 transformer。
- 风险/注意事项：
  - 目前对 transformer 的调参深度仍有限，严格说更接近“当前实现与预算下未胜出”。

### Finding 4.3 TabPFN 是很强的外部基线，但它强在 accuracy / AUC，不强在 macro-F1

- 主要数字：
  - TabPFN：accuracy `0.7861`，AUC `0.9017`，macro-F1 `0.7131`
  - 当前最好 MLP thermometer：accuracy `0.7824`，AUC `0.8903`，macro-F1 `0.7350`
  - 旧 MultiView MLP：macro-F1 `0.7217`
- 证据文件路径：
  - `docs/advanced_architecture_metrics.md`
  - `docs/run_tabpfn_baseline.md`
  - `models/metrics/tabpfn_metrics.csv`
  - `models/metrics/multiview_application_order_encoding_tuned_summary.csv`
- 报告可写一句话：
  - TabPFN 是非常强的 foundation-style baseline，在 accuracy 和 AUC 上领先，但它没有超过最佳 MLP 的 macro-F1，因此更像“概率质量强、平衡分类较弱”的对照。
- 风险/注意事项：
  - TabPFN 目前没有项目侧显式类别处理，且存在环境/认证依赖；若在 appendix 里解释复现实验，要补充这一点。

### Finding 4.4 TabM-style MLP 是有效补充，但还没超过 MultiView Gated MLP

- 主要数字：
  - TabM-style multi-expert MLP：accuracy `0.7636`，macro-F1 `0.7186`
  - 旧 MultiView Gated MLP：macro-F1 `0.7217`
  - 当前最好 thermometer MLP：macro-F1 `0.7350`
- 证据文件路径：
  - `docs/advanced_architecture_metrics.md`
  - `models/metrics/tabm_style_comparison.csv`
  - `models/metrics/multiview_application_order_encoding_tuned_summary.csv`
- 报告可写一句话：
  - 更复杂的 multi-expert MLP 并没有超过更贴合语义分组的 MultiView Gated MLP，支持“合适的视图拆分比单纯增加架构花样更重要”。
- 风险/注意事项：
  - 这条是 side comparison，更适合写成 supporting observation。

## 5. `Enrolled` 类的结构性发现：它确实是中间态，但不是一个干净可分的窄带

### Finding 5.1 `Enrolled` 大多数不靠近 Dropout 端，但也并非简单等同于 Graduate

- 主要数字：
  - endpoint view：
    - `dropout_like` 仅 `7.2%`
    - `graduate_like` `38.8%`
    - `middle_or_ambiguous` `54.0%`
  - two-stage distance tags：
    - `dropout_side` `27.8%`
    - `enrolled_core_after_not_dropout` `43.1%`
    - `stage_b_boundary` `20.2%`
    - `graduate_side_after_not_dropout` `8.9%`
- 证据文件路径：
  - `docs/analyze_enrolled_feature_space_distance.md`
  - `models/metrics/enrolled_feature_space_distance_summary.csv`
- 报告可写一句话：
  - `Enrolled` 不是噪声标签；它在特征空间中确实位于 `Dropout` 与 `Graduate` 之间，但内部包含 dropout-side、enrolled-core、boundary、graduate-side 多个子群。
- 风险/注意事项：
  - 这是解释性 side analysis，不是主模型指标；适合用来解释 two-stage tradeoff，而不是替代预测结果。

### Finding 5.2 two-stage 的主要失败点发生在 Stage A：真实 `Enrolled` 有相当一部分先被截断到 Dropout 侧

- 主要数字：
  - 约 `27.8%` 的真实 `Enrolled` 在 Stage A 线性边界上落到 `Dropout` 侧
  - 未校准 two-stage 的 Enrolled recall 只有 `0.4748`
  - threshold tuning 后回升到 `0.6176`
- 证据文件路径：
  - `docs/analyze_enrolled_feature_space_distance.md`
  - `docs/optimize_two_stage_decision_thresholds.md`
  - `models/metrics/enrolled_feature_space_distance_summary.csv`
  - `models/metrics/two_stage_threshold_tuned_metrics.csv`
- 报告可写一句话：
  - 二阶段分解不是概念上错误，而是 Stage A 若不保守，会过早把一部分真实 `Enrolled` 判成 `Dropout`；这正是 threshold tuning 能显著提高 Enrolled recall 的原因。
- 风险/注意事项：
  - distance analysis 基于 one-hot + logistic 可解释边界，不应把百分比解释为“真实标签机制”本身。

## 6. 报告优先级排序

### Must include

1. `Application order` 编码发现：thermometer-tuned MLP 当前是最佳 MLP 结果，OOF target encoding 则最利于 AUC / Enrolled recall。
2. 维度与输入范式说明：`241` vs `228`，以及 one-hot / native categorical / embedding / raw input 的区别。
3. one-stage vs two-stage 主结论：threshold tuning 明显改善 two-stage，但总体仍略弱于 best one-stage machine ensemble。
4. 传统 ML 主结果：machine-only OOF ensemble 是当前最强 classical line；单模型里 tuned CatBoost 最强。
5. `Enrolled` 是中间态但内部异质，不应简单并入 `Dropout` 或 `Graduate`。

### Useful side analysis

1. TabPFN：accuracy / AUC 最强，但 macro-F1 不如最佳 MLP。
2. bagging 家族：Pasting / ExtraTrees / Bagging 都是合格补充，但不进第一梯队。
3. classic boosting stability：GBDT 与 HistGBDT 平均 macro-F1 几乎打平，前者略偏 Enrolled，后者更稳。
4. cross-seed soft average vs stacking：soft average 值得保留，stacking 不建议主推。
5. TabM-style / transformer 系列未超过 MultiView Gated MLP。

### Needs verification

1. thermometer-tuned MLP 是否在 cross-seed / cross-split 下仍稳定优于 scalar-tuned MLP。
2. OOF target encoding 的高 Enrolled recall 是否稳健，还是当前 split 特定现象。
3. threshold-tuned two-stage 的收益是否能跨 seed 保持，特别是与 machine-only ensemble 的 `0.003` macro-F1 差距是否稳定。
4. TabPFN 的优势是否受当前环境/版本/一次性运行状态影响。
5. transformer 系列若加大调参预算后是否仍明显落后于 MLP。
