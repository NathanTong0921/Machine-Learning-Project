# DL Model Findings

- **[low] Piecewise numerical embedding（compare_piecewise_numerical_models）**
  - 该试验把数值特征做 piecewise 分桶并接入 FT-Transformer，目的是验证“数值 tokenizer 优化”是否能提效。
  - 实测结果不支持：`piecewise_ft_transformer` 的 held-out `macro-F1=0.6655`，明显低于 tuned FT（0.7053）和主线 Multi-View Gated MLP（0.7217）。
  - 结论：该方向在本数据上贡献有限，属于低优先级排查项，优先级标记为 **[low]**。

- **[medium] MLP vs FT-Transformer（compare_advanced_models）**
  - 同一数据划分下对比了 `Multi-View Gated MLP` 与 `FT-Transformer`，并记录混淆矩阵和权重。
  - 当前结果：MLP 的 `accuracy=0.7575`, `macro-F1=0.7217`, `AUC=0.8925`；FT-Transformer 为 `accuracy=0.7515`, `macro-F1=0.7053`, `AUC=0.8771`。
  - 结论：FT-Transformer 在本轮配置中仍落后于 MLP，优势仅是结构更新可复现；属于继续调参但非当前主推方向，标记为 **[medium]**。

- **[low] View-wise FT attention pooling（compare_attention_pooled_viewwise_transformers）**
  - 对比 `FT-Transformer`、`View-Wise Gated FT`（mean）与 `View-Wise Gated FT`（attention），并对齐同一划分。
  - 当前结果：`viewwise` mean `macro-F1=0.7066`，attention `macro-F1=0.7074`，仅有小幅提升；AUC 从 `0.8875` 降至 `0.8818`。
  - 结论：attention pooling 对齐了轻微 gain，但尚不足以改变主导结论；当前最强仍是 Multi-View Gated MLP，优先级标记 **[low]**。

- **[low] FT-Transformer stability check（check_ft_transformer_stability）**
  - 本文件仅描述了 FT-Transformer 的随机种子稳定性检查流程（seed=42/62/82），参数固定为 `d_token=48, n_blocks=2, lr=0.0008, attention_dropout=0.20, ff_dropout=0.20`。
  - 目的在于监测 `macro-F1` 的均值与标准差，评估模型对初始化/随机性的敏感性；若标准差偏大则说明单次优势不稳定，不宜作为主推模型依据。
  - 该说明与脚本层面关系更强，结论性信息较弱，归为低优先级补充项。  

- **[low] View-Wise Gated FT stability check（check_viewwise_gated_ft_stability）**
  - 使用 seed=42/62/82 测了当前最佳 View-Wise Gated FT（`d_token=48, n_blocks=2, lr=0.0008, attention_dropout=0.20, ff_dropout=0.20`）的稳定性。
  - 已记录汇总：`mean macro-F1=0.6967`, `std macro-F1=0.0071`, `mean accuracy=0.7294`, `mean AUC=0.8816`。
  - 结论：波动不大、但均值仍低于主线优势区域；该变体可作为次优方向，不能替代当前主线。  

- **[low] Structured tabular transformers（compare_structured_tabular_transformers）**
  - 比较对象为 `Multi-View Gated MLP`、调参 FT-Transformer 与 `view-aware FT-Transformer`（加入 periodic tokenization + view/type/column embedding）。
  - 本次结果：`view_aware_ft_transformer` 为 `acc=0.7229`, `macro-F1=0.6870`, `AUC=0.8814`，明显低于 `ft_transformer_tuned (acc=0.7515, macro-F1=0.7053)` 与 `multiview_gated_mlp_best (acc=0.7575, macro-F1=0.7217)`。
  - 结论：结构化 token/embedding 改造未带来预期收益；当前方向可保留为探索线，主推模型不变。  

- **[medium] TabM-style multi-expert MLP（compare_tabm_style_models）**
  - 对比了 `Multi-View Gated MLP` 与本地实现的 `TabM-style multi-expert MLP`，二者输出结构与训练分支接近。
  - 本次结果：`tabm_style_multiexpert_mlp` `acc=0.7636`, `macro-F1=0.7186`, `AUC=0.8888`；`multiview_gated_mlp_best` `acc=0.7575`, `macro-F1=0.7217`, `AUC=0.8925`。
  - 结论：multi-expert 在 accuracy 上有小幅领先，但 macro-F1/AUC 不及主线，说明“多 expert 思路有效但当前实现还没完全收敛为主推方案”。  

- **[medium] View-Wise Gated FT-Transformer（compare_viewwise_gated_transformers）**
  - 引入 view-wise token pooling + gated fusion，比较 `ft_transformer_tuned`、`viewwise_gated_ft_transformer` 与 `multiview_gated_mlp_best`。
  - 结果：`viewwise_gated_ft_transformer` `acc=0.7417`, `macro-F1=0.7066`, `AUC=0.8875`；`ft_transformer_tuned` `acc=0.7515`, `macro-F1=0.7053`, `AUC=0.8771`；`multiview_gated_mlp_best` `acc=0.7575`, `macro-F1=0.7217`, `AUC=0.8925`。
  - 结论：`viewwise` 在 macro-F1 上有轻微提升，说明方向有潜力，但仍落后于主线模型，不是当前主推。  

- **[medium] Conformal prediction on Supervised MoE（conformal_prediction_supervised_moe）**
  - 该实验保持 Core3 supervised error-router MoE 不变，输出从 hard label 转为 set-valued prediction，以缓解 `Enrolled` 与 `Graduate` 的结构重叠。
  - THR 与 APS 均按 OOF 概率标定，按 nominal alpha 提供覆盖率和不确定性控制；以 α=0.10 作为关键对照：
    - THR(0.10): observed coverage=0.9089，avg set size=1.4322，singleton rate=0.6032，Enrolled覆盖=0.8613，`{Enrolled,Graduate}` 比例=0.1988。
    - APS(0.10): observed coverage=0.9036，avg set size=1.8185，singleton rate=0.1182，Enrolled覆盖=0.9454，`{Enrolled,Graduate}` 比例=0.4887。
  - 结论：THR 更紧凑，APS 更适合反映歧义边界；Conformal 不直接提升硬标签 macro-F1，但对课程指标主报表是辅助、对不确定性解释与科学可读性有明显增益。  

- **[high] Multi-View Gated MLP 理论框架（theory_multiview_gated_mlp）**
  - 该文档给出的机制解释是：模型通过四个视图编码器+softmax 门控，做样本自适应的视图加权融合，即 \(z(x)=\sum_i g_i(x)h_i\) 的凸组合。
  - 机制含义是“不同学生有不同最有用视图”，比平铺式模型更能显式表达条件化特征重要性（例如晚期学业阶段与背景信息按样本动态切换）。
  - 证据链与现有结果一致：`second_semester_macro` 在 gate 上持续占优但未完全压制其他视图，说明模型兼顾了主信号与辅助先验信息。
  - 结论：这是主模型的核心理论说明，适合放入报告方法章节；但不应将其外推为“必胜于所有树模型”的普适结论，仅作为结构动机与可解释机制。  
