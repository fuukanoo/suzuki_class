from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output" / "DecisionTree"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "decision_tree.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="shift_jis")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp932")


def resolve_target_column(df: pd.DataFrame) -> str:
    candidates = ["target", "ｙ", "y", "Y", "ВЩ"]
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[-1]


def save_plot(filename: str) -> None:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved plot: %s", path)


def main() -> None:
    train_path = DATA_DIR / "train_test_2025.csv"
    df = read_csv_with_fallback(train_path)
    df["id"] = df.index

    logger.info("Original shape: %s", df.shape)
    logger.info("Preview:\n%s", df.head().to_string(index=False))

    cols_to_drop = ["education", "fnlwgt"]
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    if existing_drop_cols:
        df = df.drop(existing_drop_cols, axis=1)

    target_col = resolve_target_column(df)
    df = df.rename(columns={target_col: "target"})
    df["target"] = df["target"].astype(str).str.strip().map({"yes": 1, "no": 0})

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = num_cols.drop("target", errors="ignore")
    cat_cols = df.select_dtypes(include=["object"]).columns

    imputer_num = SimpleImputer(strategy="median")
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    imputer_cat = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    logger.info("Preprocessed shape: %s", df.shape)

    X = df.drop(["target", "id"], axis=1)
    y = df["target"]
    ids = df["id"]

    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X,
        y,
        ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    param_grid = {
        "max_depth": [5, 7, 10, 15, None],
        "min_samples_leaf": [5, 10, 20, 50],
        "criterion": ["gini", "entropy"],
        "ccp_alpha": [0.0, 0.001, 0.005, 0.01],
    }

    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        dt,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Grid search start")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    logger.info("Best params: %s", grid_search.best_params_)
    logger.info("Best CV AUC: %.4f", grid_search.best_score_)

    y_pred_prob = best_model.predict_proba(X_val)[:, 1]
    test_auc = roc_auc_score(y_val, y_pred_prob)
    logger.info("Validation AUC: %.4f", test_auc)

    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {test_auc:.4f})", color="green", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Decision Tree)")
    plt.legend(loc="lower right")
    plt.grid(True)
    save_plot("roc_curve.png")

    best_threshold = 0.5
    best_f1 = 0.0
    thresholds_range = np.arange(0.1, 0.9, 0.01)

    for th in thresholds_range:
        y_pred_bin = (y_pred_prob >= th).astype(int)
        f1 = f1_score(y_val, y_pred_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    logger.info("Optimized threshold: %.2f (F1=%.4f)", best_threshold, best_f1)

    y_pred_final = (y_pred_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred_final)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
    )
    plt.title(f"Confusion Matrix (Threshold={best_threshold:.2f})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    save_plot("confusion_matrix.png")

    logger.info("Classification report:\n%s", classification_report(y_val, y_pred_final))

    importances = best_model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances (Top 20)")
    plt.barh(range(len(indices)), importances[indices], align="center", color="green")
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.gca().invert_yaxis()
    save_plot("feature_importances.png")

    base_importance: dict[str, float] = {col: 0.0 for col in list(num_cols) + list(cat_cols)}
    cat_prefixes = {f"{col}_" for col in cat_cols}
    for name, importance in zip(feature_names, importances):
        base = None
        if name in num_cols:
            base = name
        else:
            for prefix in cat_prefixes:
                if name.startswith(prefix):
                    base = prefix[:-1]
                    break
        if base is None:
            base = name
            base_importance.setdefault(base, 0.0)
        base_importance[base] += float(importance)

    imp_df = pd.DataFrame(
        {"feature": list(base_importance.keys()), "importance": list(base_importance.values())}
    )
    imp_df = imp_df[imp_df["importance"] > 0].copy()
    if not imp_df.empty:
        imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()
    imp_df = imp_df.sort_values(by="importance", ascending=False)
    imp_output_path = OUTPUT_DIR / "feature_importance_dt.csv"
    imp_df.to_csv(imp_output_path, index=False)
    logger.info("Saved aggregated feature importance: %s", imp_output_path)

    df_valid_output = pd.DataFrame(
        {
            "id": id_val,
            "Actual": y_val,
            "DT_Probability": y_pred_prob,
            "DT_Prediction": y_pred_final,
        }
    )
    valid_output_path = OUTPUT_DIR / "valid_prediction_dt.csv"
    df_valid_output.to_csv(valid_output_path, index=False)
    logger.info("Saved validation predictions: %s", valid_output_path)

    submit_path = DATA_DIR / "submit_2025.csv"
    df_submit = read_csv_with_fallback(submit_path)
    df_submit["id"] = df_submit.index
    submit_ids = df_submit["id"]

    cols_to_exclude = ["ｙ", "probability of yes (or score)", "education", "fnlwgt"]
    df_submit_processed = df_submit.drop(
        [c for c in cols_to_exclude if c in df_submit.columns], axis=1
    )

    submit_num_cols = [c for c in num_cols if c in df_submit_processed.columns]
    submit_cat_cols = [c for c in cat_cols if c in df_submit_processed.columns]

    df_submit_processed[submit_num_cols] = imputer_num.transform(
        df_submit_processed[submit_num_cols]
    )
    df_submit_processed[submit_cat_cols] = imputer_cat.transform(
        df_submit_processed[submit_cat_cols]
    )

    df_submit_processed = pd.get_dummies(
        df_submit_processed, columns=submit_cat_cols, drop_first=True
    )

    df_submit_processed = df_submit_processed.reindex(columns=X_train.columns, fill_value=0)

    submit_probs = best_model.predict_proba(df_submit_processed)[:, 1]

    df_submit_output = pd.DataFrame(
        {
            "id": submit_ids,
            "DT_Probability": submit_probs,
            "DT_Prediction": (submit_probs >= best_threshold).astype(int),
        }
    )
    submit_output_path = OUTPUT_DIR / "test_prediction_dt.csv"
    df_submit_output.to_csv(submit_output_path, index=False)
    logger.info("Saved test predictions: %s", submit_output_path)


if __name__ == "__main__":
    main()
