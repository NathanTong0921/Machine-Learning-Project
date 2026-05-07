# Script Layout

Run scripts from the project root, for example:

```bash
python scripts/preprocessing/prepare_data.py
python scripts/boost/tune_catboost_baseline.py
python scripts/mlp/run_advanced.py
```

## Directories

| Directory | Purpose |
| --- | --- |
| `preprocessing/` | Dataset preparation and train/test split generation. |
| `baseline/` | Mixed classical baseline models used for the mandatory task. |
| `bagging/` | Bagging-style tree models, including balanced Random Forest, native decision-tree bagging, pasting-style trees, and Extra Trees. |
| `boost/` | Boosted tree models and tuning: CatBoost, LightGBM, XGBoost, and semester-feature tree experiments. |
| `feature_engineering/` | Feature-engineering experiments such as polynomial/RBF baselines and target-encoded semester features. |
| `mlp/` | MLP-family neural tabular models, including Multi-View Gated MLP and TabM-style multi-expert MLP. |
| `transformer/` | Transformer and prompt-style tabular models, including FT-Transformer variants, ViewWise transformers, Trompt-style transformer, and TabPFN. |
| `ensemble/` | Voting, stacking, OOF probability ensembles, and ensemble decision-bias optimization. |
| `clustering/` | Unsupervised clustering experiments. |
| `reporting/` | Result aggregation and report-ready summary table/figure generation. |
