"""Classical baseline models such as Logistic Regression, Trees, Ensembles, and SVM."""

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_baseline_models(random_state=42):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
            n_estimators=200,
            learning_rate=0.5,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def get_baseline_search_spaces(random_state=42):
    return {
        "logistic_regression": {
            "model": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_grid": {
                "model__C": [0.05, 0.2, 1.0, 5.0],
            },
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(random_state=random_state),
            "param_grid": {
                "model__max_depth": [4, 6, 10, None],
                "model__min_samples_leaf": [1, 2, 5],
                "model__criterion": ["gini", "entropy"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [8, 12, None],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 0.5],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "param_grid": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        },
        "adaboost": {
            "model": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=random_state),
                random_state=random_state,
            ),
            "param_grid": {
                "model__estimator__max_depth": [1, 2, 3],
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.1, 0.5, 1.0],
            },
        },
        "extra_trees": {
            "model": ExtraTreesClassifier(
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [8, 12, None],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 0.5],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
        },
        "svm_rbf": {
            "model": SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_grid": {
                "model__C": [0.5, 1.0, 2.0, 5.0],
                "model__gamma": ["scale", 0.05, 0.1, 0.2],
            },
        },
    }
