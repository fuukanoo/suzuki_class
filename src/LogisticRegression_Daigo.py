from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output" / "LogisticRegression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logistic_regression.log", encoding="utf-8"),
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
    target_col = resolve_target_column(df)
    df = df.rename(columns={target_col: "target"})
    df["id"] = df.index

    logger.info("Training data shape: %s", df.shape)

    y_raw = df["target"].astype(str).str.strip().str.lower()
    y = y_raw.replace({"no": "0", "yes": "1"}).astype(float).astype(int)

    X = df.drop(columns=["target", "id"])
    ids = df["id"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X,
        y,
        ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    logger.info("Model training complete")

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    auc_score = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)

    logger.info("AUC: %.4f", auc_score)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Classification report:\n%s", classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    save_plot("roc_curve.png")

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    for th in thresholds:
        preds = (y_pred_proba >= th).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    logger.info("Best threshold by F1: %.2f", best_threshold)

    df_valid_output = pd.DataFrame(
        {
            "id": id_val,
            "Actual": y_val,
            "LR_Probability": y_pred_proba,
            "LR_Prediction": (y_pred_proba >= best_threshold).astype(int),
        }
    )
    valid_output_path = OUTPUT_DIR / "valid_prediction_lr.csv"
    df_valid_output.to_csv(valid_output_path, index=False)
    logger.info("Saved validation predictions: %s", valid_output_path)

    submit_path = DATA_DIR / "submit_2025.csv"
    submit_df = read_csv_with_fallback(submit_path)
    submit_df["id"] = submit_df.index

    target_col_submit = submit_df.columns[-2]
    prob_col_submit = submit_df.columns[-1]

    X_submit = submit_df.drop(columns=[target_col_submit, prob_col_submit], errors="ignore")
    X_submit = X_submit.reindex(columns=X.columns, fill_value=np.nan)

    submit_pred_class = model.predict(X_submit)
    submit_pred_proba = model.predict_proba(X_submit)[:, 1]

    df_submit_output = pd.DataFrame(
        {
            "id": submit_df["id"],
            "LR_Probability": submit_pred_proba,
            "LR_Prediction": (submit_pred_proba >= best_threshold).astype(int),
        }
    )
    submit_output_path = OUTPUT_DIR / "test_prediction_lr.csv"
    df_submit_output.to_csv(submit_output_path, index=False)
    logger.info("Saved test predictions: %s", submit_output_path)

    submit_output = submit_df.copy()
    submit_output[target_col_submit] = np.where(submit_pred_class == 1, "yes", "no")
    submit_output[prob_col_submit] = submit_pred_proba

    output_filename = OUTPUT_DIR / "submit_prediction_logistic.csv"
    submit_output.to_csv(output_filename, index=False)
    logger.info("Saved submission file: %s", output_filename)

    try:
        feature_names = numeric_features.tolist() + (
            model.named_steps["preprocessor"]
            .transformers_[1][1]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_features)
            .tolist()
        )
        coefficients = model.named_steps["classifier"].coef_[0]
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        coef_df["abs_coef"] = coef_df["coefficient"].abs()
        top5 = coef_df.sort_values(by="abs_coef", ascending=False).head(5)
        logger.info("Top 5 coefficients:\n%s", top5.to_string(index=False))

        base_importance: dict[str, float] = {
            col: 0.0 for col in list(numeric_features) + list(categorical_features)
        }
        cat_prefixes = {f"{col}_" for col in categorical_features}
        for name, coef in zip(feature_names, coefficients):
            base = None
            if name in numeric_features:
                base = name
            else:
                for prefix in cat_prefixes:
                    if name.startswith(prefix):
                        base = prefix[:-1]
                        break
            if base is None:
                base = name
                base_importance.setdefault(base, 0.0)
            base_importance[base] += abs(float(coef))

        imp_df = pd.DataFrame(
            {"feature": list(base_importance.keys()), "importance": list(base_importance.values())}
        )
        imp_df = imp_df[imp_df["importance"] > 0].copy()
        if not imp_df.empty:
            imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()
        imp_df = imp_df.sort_values(by="importance", ascending=False)
        imp_output_path = OUTPUT_DIR / "feature_importance_lr.csv"
        imp_df.to_csv(imp_output_path, index=False)
        logger.info("Saved aggregated feature importance: %s", imp_output_path)
    except Exception as exc:
        logger.warning("Coefficient extraction failed: %s", exc)


if __name__ == "__main__":
    main()
