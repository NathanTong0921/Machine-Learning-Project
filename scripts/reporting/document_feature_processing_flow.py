"""Generate a Markdown report for feature processing flow.

The report answers which columns become one-hot, which columns stay numeric,
and what each model family receives at fit time under the current code state.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import prepare_ft_transformer_data, prepare_multiview_data
from src.config import CATEGORICAL_COLUMNS, FEATURE_GROUPS, PROCESSED_DIR, TARGET_COL
from src.preprocessing import build_preprocessor, split_feature_columns


OUTPUT_PATH = PROJECT_ROOT / "docs" / "feature_processing_flow.md"
REFERENCE_SEED = 62


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def bullet_list(values):
    return "\n".join(f"- `{value}`" for value in values)


def categorical_count_table(train_df, test_df, categorical_features):
    rows = []
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    for col in categorical_features:
        rows.append(
            [
                f"`{col}`",
                train_df[col].nunique(dropna=True),
                test_df[col].nunique(dropna=True),
                all_df[col].nunique(dropna=True),
            ]
        )
    return markdown_table(["feature", "train unique", "test unique", "all unique"], rows)


def sklearn_onehot_summary(train_df, numeric_features, categorical_features):
    X_train = train_df.drop(columns=[TARGET_COL])
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train)
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    rows = []
    total_cat_dim = 0
    for col, categories in zip(categorical_features, onehot.categories_):
        cat_dim = len(categories)
        total_cat_dim += cat_dim
        rows.append([f"`{col}`", cat_dim])
    return {
        "rows": rows,
        "numeric_dim": len(numeric_features),
        "categorical_dim": total_cat_dim,
        "total_dim": len(numeric_features) + total_cat_dim,
    }


def multiview_summary(train_df, test_df):
    bundle = prepare_multiview_data(train_df, test_df, random_state=REFERENCE_SEED)
    rows = []
    for view_name, columns in FEATURE_GROUPS.items():
        categorical = [col for col in columns if col in CATEGORICAL_COLUMNS]
        numeric = [col for col in columns if col not in CATEGORICAL_COLUMNS]
        rows.append(
            [
                f"`{view_name}`",
                bundle.view_dims[view_name],
                len(categorical),
                ", ".join(f"`{col}`" for col in categorical) if categorical else "-",
                len(numeric),
                ", ".join(f"`{col}`" for col in numeric) if numeric else "-",
            ]
        )
    return bundle, rows


def ft_summary(train_df, test_df):
    bundle = prepare_ft_transformer_data(train_df, test_df, random_state=REFERENCE_SEED)
    return bundle


def write_report():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_columns = train_df.drop(columns=[TARGET_COL]).columns.tolist()
    numeric_features, categorical_features = split_feature_columns(feature_columns)
    sklearn_summary = sklearn_onehot_summary(train_df, numeric_features, categorical_features)
    multiview_bundle, multiview_rows = multiview_summary(train_df, test_df)
    ft_bundle = ft_summary(train_df, test_df)

    legacy_multiview_total = sum(multiview_bundle.view_dims.values()) + 7
    current_multiview_total = sum(multiview_bundle.view_dims.values())

    parts = []
    parts.append("# Feature Processing Flow\n")
    parts.append(
        "这份文档由 `scripts/reporting/document_feature_processing_flow.py` 自动生成，"
        "反映当前代码和 `data/processed/train.csv` / `data/processed/test.csv` 的最新状态。\n"
    )
    parts.append("## 数据入口\n")
    parts.append(
        markdown_table(
            ["artifact", "rows", "columns", "feature columns", "target"],
            [
                ["`data/processed/train.csv`", len(train_df), train_df.shape[1], len(feature_columns), f"`{TARGET_COL}`"],
                ["`data/processed/test.csv`", len(test_df), test_df.shape[1], len(feature_columns), f"`{TARGET_COL}`"],
            ],
        )
    )
    parts.append("\n`scripts/preprocessing/prepare_data.py` 不做 one-hot，也不做标准化；它只读取 raw CSV、清理列名/`Target`、做 stratified train/test split 并保存 processed CSV。\n")

    parts.append("## 当前语义列划分\n")
    parts.append(f"当前共有 `{len(categorical_features)}` 个 semantic categorical 特征：\n")
    parts.append(bullet_list(categorical_features))
    parts.append(f"\n当前共有 `{len(numeric_features)}` 个 numeric/ordinal/continuous 特征：\n")
    parts.append(bullet_list(numeric_features))
    parts.append(
        "\n注意：`Application order` 当前不在 `CATEGORICAL_COLUMNS` 中，因此按 numeric/ordinal 处理。"
        "这与 PDF Table 1 的 `Numeric/ordinal` 标注一致。\n"
    )
    parts.append(
        "补充：对线性/距离/MLP 模型，把 `Application order` 作为单个 scaled scalar 会隐含等距 ordinal 假设。"
        "已在 `scripts/mlp/compare_application_order_encodings_multiview.py` 中测试 thermometer 和 OOF target encoding；"
        "进一步在 `scripts/mlp/search_application_order_encoding_multiview.py` 中调参后，thermometer encoding "
        "超过 tuned scalar MultiView MLP 的 macro-F1，OOF target encoding 则给出更高 AUC 和 `Enrolled` recall。\n"
    )

    parts.append("## Categorical Unique Count\n")
    parts.append(categorical_count_table(train_df, test_df, categorical_features))
    parts.append("\n这些 unique count 解释了 one-hot 维度。实际模型 fit 时只用训练 split fit encoder，因此 test 中未见过的类别由 `handle_unknown='ignore'` 或 native categorical 机制处理。\n")

    parts.append("## Classical ML: sklearn Pipeline One-Hot\n")
    parts.append(
        "适用脚本/模型：`scripts/baseline/run_baselines.py`、`scripts/bagging/*`、"
        "`scripts/boost/search_classic_boosting_models.py`、`scripts/ensemble/tune_machine_only_ensemble.py` 中的 "
        "Logistic Regression / SVM / AdaBoost / GBDT / HistGradientBoosting / Bagging decision tree / Extra Trees 等 sklearn 成员。\n"
    )
    parts.append("处理方式：\n")
    parts.append("- numeric features: `SimpleImputer(strategy='median') -> StandardScaler()`\n")
    parts.append("- categorical features: `SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore')`\n")
    parts.append("- 最后喂给 sklearn estimator 的是拼接后的 tabular matrix。\n")
    parts.append(
        markdown_table(
            ["part", "dimension fitted on processed train"],
            [
                ["numeric scaled", sklearn_summary["numeric_dim"]],
                ["categorical one-hot", sklearn_summary["categorical_dim"]],
                ["total model input", sklearn_summary["total_dim"]],
            ],
        )
    )
    parts.append("\n按列展开：\n")
    parts.append(markdown_table(["categorical feature", "one-hot dimension on processed train"], sklearn_summary["rows"]))

    parts.append("## Logistic Regression And AdaBoost Status\n")
    parts.append(
        markdown_table(
            ["question", "current status", "where"],
            [
                [
                    "普通 Logistic Regression 是否有",
                    "有。作为 baseline 使用 one-hot + scaled numeric 后接 `LogisticRegression(class_weight='balanced')`；grid-search 也调过 `C`。",
                    "`src/baselines.py`, `scripts/baseline/run_baselines.py`, `scripts/ensemble/search_and_ensemble_baselines.py`",
                ],
                [
                    "特征工程 Logistic Regression 是否有",
                    "有。包括 polynomial features、RBF Nystroem map、target encoding + semester features。",
                    "`scripts/feature_engineering/run_feature_engineered_baselines.py`, `scripts/feature_engineering/target_encoding_semester_features.py`",
                ],
                [
                    "Logistic Regression stacking 是否有",
                    "有。OOF probability stacking 用 `LogisticRegression(max_iter=5000, class_weight='balanced')` 作为 meta learner。",
                    "`scripts/ensemble/compare_cross_seed_stacking.py`, `scripts/ensemble/stack_oof_probability_ensemble.py`",
                ],
                [
                    "AdaBoost with Logistic Regression base estimator 是否有",
                    "没有。当前 AdaBoost 只用 shallow `DecisionTreeClassifier` / stump-style base estimator；没有跑 LogisticRegression 作为 AdaBoost base estimator。",
                    "`src/baselines.py`, `scripts/boost/search_classic_boosting_models.py`, `scripts/ensemble/tune_machine_only_ensemble.py`",
                ],
            ],
        )
    )
    parts.append(
        "\n说明：`gbdt_log_loss` 覆盖的是 log-loss gradient boosting / LogitBoost-style 对照，"
        "不是 `AdaBoostClassifier(estimator=LogisticRegression(...))`。\n"
    )

    parts.append("## Classical ML: Native Categorical Tree Models\n")
    parts.append(
        markdown_table(
            ["model family", "representative scripts", "categorical treatment", "numeric treatment", "final model input"],
            [
                [
                    "CatBoost",
                    "`scripts/boost/tune_catboost_baseline.py`, ensemble CatBoost members",
                    "`CATEGORICAL_COLUMNS` cast to `str`, passed via `cat_features` indices",
                    "raw numeric values, no sklearn scaling",
                    "36-column DataFrame; CatBoost handles categorical internally",
                ],
                [
                    "LightGBM",
                    "`scripts/boost/search_hist_column_sampling.py`, ensemble LightGBM members",
                    "`CATEGORICAL_COLUMNS` cast to pandas `category`, passed via `categorical_feature`",
                    "raw numeric values, no sklearn scaling",
                    "36-column DataFrame; LightGBM handles categorical splits internally",
                ],
                [
                    "XGBoost hist",
                    "`scripts/boost/search_hist_column_sampling.py`, ensemble XGBoost hist members",
                    "`CATEGORICAL_COLUMNS` cast to pandas `category`, `enable_categorical=True`",
                    "raw numeric values, no sklearn scaling",
                    "36-column DataFrame; XGBoost hist handles categorical features natively",
                ],
            ],
        )
    )
    parts.append("\n这里没有 one-hot 展开；`max_bin`、`column_fraction` 等参数作用在原始 36 列特征空间上。\n")

    parts.append("## Advanced DL: MultiView Gated MLP / TabM-Style MLP\n")
    parts.append(
        "适用脚本：`scripts/mlp/run_advanced.py`、`scripts/mlp/retrain_multiview_current_preprocess.py`、"
        "`scripts/mlp/search_multiview_current_preprocess.py`、`scripts/mlp/compare_tabm_style_models.py` 等。\n"
    )
    parts.append(
        f"处理方式：每个 view 单独 fit 一个 `ColumnTransformer`。下表使用 reference seed `{REFERENCE_SEED}`，"
        "也就是当前 current-preprocess 搜索和 best checkpoint 的主要口径。\n"
    )
    parts.append("- view 内 numeric: median impute + standard scaling\n")
    parts.append("- view 内 categorical: most-frequent impute + dense one-hot\n")
    parts.append("- 最后喂给模型的是多个 tensor：`enrollment_profile`, `family_finance`, `first_semester`, `second_semester_macro`\n")
    parts.append(
        markdown_table(
            ["view", "input dim", "# categorical cols", "categorical cols", "# numeric cols", "numeric cols"],
            multiview_rows,
        )
    )
    parts.append(
        f"\n在 reference seed `{REFERENCE_SEED}` 下，当前 MultiView total dim 是 `{current_multiview_total}`。其中 `enrollment_profile=80` 只是单个 view，"
        "不是全量 one-hot 宽度。旧 checkpoint 的 `enrollment_profile=87` 来自把 `Application order` 当 categorical，"
        f"旧兼容 total dim 是 `{legacy_multiview_total}`。\n"
    )
    parts.append(
        "注意：`prepare_multiview_data()` 会先把 processed train 再切 internal train/val，"
        "并只在 internal train 上 fit one-hot encoder；所以 view dims 依赖 seed/fold。"
        "例如 default seed `42` 不一定等于 reference seed `62` 的 `228`。\n"
    )

    parts.append("## Advanced DL: FTTransformer / View-Wise / Piecewise / Trompt-Style Transformer\n")
    parts.append(
        "适用脚本：`scripts/transformer/compare_advanced_models.py`、`compare_viewwise_gated_transformers.py`、"
        "`compare_piecewise_numerical_models.py`、`compare_trompt_style_models.py` 等。\n"
    )
    parts.append("处理方式：\n")
    parts.append("- categorical features 不做 one-hot；按训练 split 建立 category -> integer id，未知类别映射到 0。\n")
    parts.append("- numeric features 做 median fill，然后按训练 split mean/std 标准化。\n")
    parts.append("- 最后喂给模型两个 tensor：`x_cat` 和 `x_num`。\n")
    parts.append(
        "- Piecewise FT 的 numeric 输入仍是标准化后的 `x_num`，但分箱边界由原始 internal train numeric values 计算；"
        "不要把它描述成在标准化值上直接分箱。\n"
    )
    parts.append(
        markdown_table(
            ["tensor", "shape per sample", "meaning"],
            [
                ["`x_cat`", len(ft_bundle.cat_feature_names), "17 个 categorical id，进入 embedding/tokenizer"],
                ["`x_num`", len(ft_bundle.num_feature_names), "19 个标准化 numeric/ordinal/continuous value"],
            ],
        )
    )
    parts.append("categorical cardinalities 基于 internal train split 统计，含 unknown id 0：\n")
    parts.append(
        markdown_table(
            ["categorical feature", "cardinality incl. unknown"],
            [[f"`{col}`", card] for col, card in zip(ft_bundle.cat_feature_names, ft_bundle.cat_cardinalities)],
        )
    )

    parts.append("## TabPFN\n")
    parts.append(
        "适用脚本：`scripts/transformer/run_tabpfn_baseline.py`。当前代码没有显式 one-hot、scaling 或 categorical cast；"
        "直接把 36-column pandas DataFrame 传入 `TabPFNClassifier.fit(X_train, y_train)`。因此：\n"
    )
    parts.append("- categorical integer codes 仍作为原始列值进入 TabPFN。\n")
    parts.append("- numeric/ordinal/continuous features 也作为原始列值进入 TabPFN。\n")
    parts.append("- 具体内部归一化/编码由 TabPFN 包自身处理，项目代码层面没有手工展开。\n")

    parts.append("## Summary: What Is Fed Into Each Model\n")
    parts.append(
        markdown_table(
            ["branch", "one-hot?", "scaled numeric?", "native categorical?", "final input"],
            [
                ["sklearn ML baselines/bagging/classic boosting", "yes, 17 categorical columns", "yes", "no", f"{sklearn_summary['total_dim']}-dim transformed matrix"],
                ["CatBoost", "no", "no", "yes, `str` cat columns", "raw 36-column DataFrame + cat feature indices"],
                ["LightGBM", "no", "no", "yes, pandas `category`", "raw 36-column DataFrame + categorical_feature names"],
                ["XGBoost hist", "no", "no", "yes, pandas `category`", "raw 36-column DataFrame with `enable_categorical=True`"],
                ["MultiView Gated MLP / TabM-style", "yes, within each view", "yes, within each view", "no", f"four dense view tensors, seed {REFERENCE_SEED} total dim 228"],
                ["FTTransformer-style advanced models", "no", "yes", "embedding ids", "`x_cat` 17 ids + `x_num` 19 floats"],
                ["TabPFN", "no explicit project-side one-hot", "no explicit project-side scaling", "not explicitly declared", "raw 36-column DataFrame"],
            ],
        )
    )

    parts.append("## Source Files To Check\n")
    parts.append("- `src/config.py`: semantic categorical columns and MultiView feature groups\n")
    parts.append("- `src/preprocessing.py`: sklearn one-hot preprocessing\n")
    parts.append("- `src/advanced_models.py`: MultiView and FT-style preprocessing\n")
    parts.append("- `scripts/boost/tune_catboost_baseline.py`: CatBoost native categorical path\n")
    parts.append("- `scripts/boost/search_hist_column_sampling.py`: LightGBM/XGBoost native categorical path\n")
    parts.append("- `scripts/ensemble/tune_machine_only_ensemble.py`: mixed machine-only ensemble preprocessing\n")
    parts.append("- `scripts/transformer/run_tabpfn_baseline.py`: raw DataFrame TabPFN input\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"saved feature processing flow: {OUTPUT_PATH}")


if __name__ == "__main__":
    write_report()
