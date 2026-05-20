from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_data import (
    TARGET_COLUMN,
    build_preprocessor,
    load_or_prepare_processed_data,
    make_train_test_split,
)
from paths import (
    FEATURE_IMPORTANCE_PATH,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    REPORTS_DIR,
    VISUALS_DIR,
    ensure_project_dirs,
)


def _try_make_xgboost(scale_pos_weight: float, random_state: int) -> tuple[Any, dict[str, list[Any]]] | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    params = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 4],
        "model__learning_rate": [0.03, 0.08],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }
    return estimator, params


def model_search_spaces(
    y_train: np.ndarray,
    random_state: int = 42,
    include_xgboost: bool = True,
) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    churn_count = max(int((y_train == 1).sum()), 1)
    non_churn_count = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = non_churn_count / churn_count

    spaces: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            ),
            {
                "model__C": [0.05, 0.1, 1.0, 3.0],
                "model__penalty": ["l1", "l2"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [20, 50, 100],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [6, 10, None],
                "model__min_samples_leaf": [10, 25],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
    }

    if include_xgboost:
        xgb_space = _try_make_xgboost(scale_pos_weight, random_state)
        if xgb_space is not None:
            spaces["XGBoost"] = xgb_space

    return spaces


def evaluate_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float | int]:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
    random_state: int = 42,
    include_xgboost: bool = True,
    scoring_metric: str = "recall",
) -> tuple[pd.DataFrame, dict[str, Pipeline], Any]:
    """Train tuned classification models and return metrics plus fitted estimators."""
    split = make_train_test_split(df, random_state=random_state)
    preprocessor = build_preprocessor(split.numeric_features, split.categorical_features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    search_spaces = model_search_spaces(
        split.y_train,
        random_state=random_state,
        include_xgboost=include_xgboost,
    )

    for model_name, (estimator, param_grid) in search_spaces.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1_score": "f1",
                "roc_auc": "roc_auc",
            },
            refit=scoring_metric,
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(split.X_train, split.y_train)

        best_model = grid_search.best_estimator_
        metrics = evaluate_classifier(best_model, split.X_test, split.y_test)
        results.append(
            {
                "model": model_name,
                "best_cv_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
                **metrics,
            }
        )
        fitted_models[model_name] = best_model

    metrics_df = pd.DataFrame(results).sort_values(
        by=["recall", "roc_auc", "f1_score"],
        ascending=False,
    )
    return metrics_df, fitted_models, split


def select_best_model(metrics_df: pd.DataFrame, fitted_models: dict[str, Pipeline]) -> tuple[str, Pipeline]:
    best_model_name = metrics_df.iloc[0]["model"]
    return str(best_model_name), fitted_models[str(best_model_name)]


def get_feature_names(model: Pipeline) -> np.ndarray:
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    feature_names = get_feature_names(model)
    classifier = model.named_steps["model"]

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(classifier, "coef_"):
        importance_values = classifier.coef_[0]
        importance_type = "logistic_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "absolute_importance", "importance_type"])

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
            "absolute_importance": np.abs(importance_values),
            "importance_type": importance_type,
        }
    )
    return importance_df.sort_values("absolute_importance", ascending=False).reset_index(drop=True)


def save_model_artifacts(
    metrics_df: pd.DataFrame,
    fitted_models: dict[str, Pipeline],
    split: Any,
) -> dict[str, Path | str]:
    ensure_project_dirs()
    best_model_name, best_model = select_best_model(metrics_df, fitted_models)

    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    feature_importance = extract_feature_importance(best_model)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    y_pred = best_model.predict(split.X_test)
    y_score = best_model.predict_proba(split.X_test)[:, 1]
    predictions = split.X_test.copy()
    predictions["actual_churn"] = split.label_encoder.inverse_transform(split.y_test)
    predictions["predicted_churn"] = split.label_encoder.inverse_transform(y_pred)
    predictions["churn_probability"] = y_score
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    joblib.dump(
        {
            "model_name": best_model_name,
            "model": best_model,
            "target_column": TARGET_COLUMN,
            "label_classes": split.label_encoder.classes_.tolist(),
            "feature_columns": split.feature_columns,
            "metrics": metrics_df.to_dict(orient="records"),
        },
        MODEL_PATH,
    )

    return {
        "best_model_name": best_model_name,
        "model_path": MODEL_PATH,
        "metrics_path": MODEL_METRICS_PATH,
        "predictions_path": PREDICTIONS_PATH,
        "feature_importance_path": FEATURE_IMPORTANCE_PATH,
    }


def plot_roc_curves(
    fitted_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_path: Path | None = None,
) -> Path:
    ensure_project_dirs()
    output_path = output_path or (VISUALS_DIR / "roc_curves.png")

    plt.figure(figsize=(9, 6))
    for model_name, model in fitted_models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_best_confusion_matrix(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_path: Path | None = None,
) -> Path:
    ensure_project_dirs()
    output_path = output_path or (VISUALS_DIR / "best_model_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Best Model Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: Path | None = None,
    top_n: int = 20,
) -> Path:
    ensure_project_dirs()
    output_path = output_path or (VISUALS_DIR / "feature_importance_top20.png")

    top_features = feature_importance.head(top_n).copy()
    top_features["feature"] = top_features["feature"].str.replace("categorical__", "", regex=False)
    top_features["feature"] = top_features["feature"].str.replace("numeric__", "", regex=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_features, y="feature", x="absolute_importance", color="#2f6f9f")
    plt.xlabel("Absolute Importance")
    plt.ylabel("")
    plt.title(f"Top {top_n} Churn Prediction Drivers")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def run_modeling_pipeline(
    raw_csv: Path | str | None = None,
    include_xgboost: bool = True,
) -> dict[str, Path | str]:
    df = load_or_prepare_processed_data(raw_csv=raw_csv, force=raw_csv is not None)
    metrics_df, fitted_models, split = train_and_evaluate_models(
        df,
        include_xgboost=include_xgboost,
    )
    artifacts = save_model_artifacts(metrics_df, fitted_models, split)
    best_model_name, best_model = select_best_model(metrics_df, fitted_models)

    plot_roc_curves(fitted_models, split.X_test, split.y_test)
    plot_best_confusion_matrix(best_model, split.X_test, split.y_test)
    feature_importance = extract_feature_importance(best_model)
    if not feature_importance.empty:
        plot_feature_importance(feature_importance)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Model comparison:")
    print(metrics_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]])
    print(f"Best model selected by recall and ROC-AUC: {best_model_name}")
    return artifacts
