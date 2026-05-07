# Student Dropout Course Project

课程项目工作区，基于 UCI `Predict Students' Dropout and Academic Success` 数据集，覆盖：

- 数据预处理
- t-SNE 可视化
- 聚类分析
- 分类建模与评估
- 开放探索模型

## 推荐工作流

1. 原始数据与项目说明放在 `data/raw/` 和 `docs/`
2. 清洗与特征工程结果放在 `data/interim/` 和 `data/processed/`
3. 主要执行入口使用 `py` 脚本，放在 `scripts/`
4. 可复用代码放在 `src/`
5. 图表、报告、slides、模型结果分别放在 `reports/`、`presentation/`、`models/`
6. 最终提交材料统一整理到 `submission/`

## 目录

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── scripts/
├── src/
├── reports/
│   └── figures/
├── presentation/
├── models/
│   ├── saved_models/
│   └── metrics/
├── submission/
└── idea-stage/
```

## 当前文件落位

- 原始数据：`data/raw/data.csv`
- 课程 PDF：`docs/DSAA2011-26sp-project.pdf`
- 选题建议：`docs/IDEA_REPORT.md`
- 项目声明与建议：`docs/DECLARATION_AND_TIPS.md`

## 下一步建议

- 在 `src/preprocessing.py` 完成列名清洗、类别列/数值列划分和预处理 pipeline
- 从 `scripts/prepare_data.py` 开始跑数据检查与拆分
- 从 `scripts/run_baselines.py` 开始跑 baseline
- 在 `src/advanced_models.py` 实现 `Multi-View Gated MLP`

## 分工

- 预处理：钟卓熹
- visualization + t-SNE：郭子贤
- 聚类：童照
- 监督学习：黄思铭
- model evaluation + OPEN Ended Exploration：李佳锴、钟卓熹、以上所有做完自己负责部分的人
