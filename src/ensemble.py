from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "output"
ENSEMBLE_DIR = OUTPUT_ROOT / "Ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_DIRS = {
    "dt": OUTPUT_ROOT / "DecisionTree",
    "lr": OUTPUT_ROOT / "LogisticRegression",
    "gbm": OUTPUT_ROOT / "LightGBM",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ENSEMBLE_DIR / "ensemble.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def save_plot(filename: str) -> None:
    path = ENSEMBLE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved plot: %s", path)


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="shift_jis")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp932")


def detect_prob_column(df: pd.DataFrame) -> str:
    prob_cols = [c for c in df.columns if "prob" in c.lower() or "score" in c.lower()]
    if not prob_cols:
        raise ValueError("Probability column not found")
    return prob_cols[0]


def detect_actual_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() == "actual":
            return col
    raise ValueError("Actual column not found")


def load_valid_predictions(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    prob_col = detect_prob_column(df)
    actual_col = detect_actual_column(df)
    return df[["id", actual_col, prob_col]].rename(
        columns={actual_col: "Actual", prob_col: f"prob_{model_name}"}
    )


def load_test_predictions(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    prob_col = detect_prob_column(df)
    return df[["id", prob_col]].rename(columns={prob_col: f"prob_{model_name}"})


def load_feature_importance(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "feature" not in df.columns or "importance" not in df.columns:
        raise ValueError(f"Invalid importance file format: {path}")
    return df.set_index("feature")["importance"]


def main() -> None:
    logger.info("Loading validation predictions")
    valid_files = {
        "dt": "valid_prediction_dt.csv",
        "lr": "valid_prediction_lr.csv",
        "gbm": "valid_prediction_gbm.csv",
    }

    dfs_valid: dict[str, pd.DataFrame] = {}
    for model_name, file_name in valid_files.items():
        path = MODEL_OUTPUT_DIRS[model_name] / file_name
        dfs_valid[model_name] = load_valid_predictions(path, model_name)

    df_ensemble_valid = dfs_valid["dt"]
    for model in ["lr", "gbm"]:
        df_ensemble_valid = df_ensemble_valid.merge(
            dfs_valid[model][["id", f"prob_{model}"]],
            on="id",
            how="inner",
        )

    logger.info("Merged validation rows: %s", len(df_ensemble_valid))

    logger.info("Correlation check")
    corr_cols = ["prob_dt", "prob_lr", "prob_gbm"]
    corr_matrix = df_ensemble_valid[corr_cols].corr()
    logger.info("\n%s", corr_matrix)

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".3f")
    plt.title("Prediction Correlation Matrix")
    save_plot("correlation_heatmap.png")

    logger.info("Splitting data for optimization/evaluation")
    df_opt, df_eval = train_test_split(
        df_ensemble_valid,
        test_size=0.5,
        random_state=42,
        stratify=df_ensemble_valid["Actual"],
    )

    logger.info("Optimizing weights")
    best_auc = 0.0
    best_weights = {"dt": 0.0, "lr": 0.0, "gbm": 0.0}
    step = 0.05
    range_vals = np.arange(0, 1.01, step)

    for w_dt in range_vals:
        for w_lr in range_vals:
            w_gbm = 1.0 - (w_dt + w_lr)
            if w_gbm >= -0.001:
                ensemble_prob = (
                    df_opt["prob_dt"] * w_dt
                    + df_opt["prob_lr"] * w_lr
                    + df_opt["prob_gbm"] * w_gbm
                )
                score = roc_auc_score(df_opt["Actual"], ensemble_prob)

                if score > best_auc:
                    best_auc = score
                    best_weights = {"dt": w_dt, "lr": w_lr, "gbm": w_gbm}

    logger.info(
        "Best weights (opt set): DT=%.2f, LR=%.2f, GBM=%.2f",
        best_weights["dt"],
        best_weights["lr"],
        best_weights["gbm"],
    )

    logger.info("Optimizing threshold")
    opt_probs = (
        df_opt["prob_dt"] * best_weights["dt"]
        + df_opt["prob_lr"] * best_weights["lr"]
        + df_opt["prob_gbm"] * best_weights["gbm"]
    )

    best_threshold = 0.5
    best_f1_opt = 0.0
    thresholds = np.arange(0.2, 0.8, 0.01)

    for th in thresholds:
        preds = (opt_probs >= th).astype(int)
        score = f1_score(df_opt["Actual"], preds)
        if score > best_f1_opt:
            best_f1_opt = score
            best_threshold = th

    logger.info("Optimized threshold: %.2f", best_threshold)

    logger.info("Evaluating on hold-out")
    eval_probs = (
        df_eval["prob_dt"] * best_weights["dt"]
        + df_eval["prob_lr"] * best_weights["lr"]
        + df_eval["prob_gbm"] * best_weights["gbm"]
    )

    final_valid_auc = roc_auc_score(df_eval["Actual"], eval_probs)
    final_valid_preds = (eval_probs >= best_threshold).astype(int)
    final_valid_f1 = f1_score(df_eval["Actual"], final_valid_preds)

    logger.info("Evaluation AUC: %.5f", final_valid_auc)
    logger.info("Evaluation F1 : %.5f", final_valid_f1)

    logger.info("Classification report:\n%s", classification_report(df_eval["Actual"], final_valid_preds))

    cm = confusion_matrix(df_eval["Actual"], final_valid_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
    )
    plt.title(f"Confusion Matrix (Threshold={best_threshold:.2f})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    save_plot("confusion_matrix.png")

    fpr, tpr, _ = roc_curve(df_eval["Actual"], eval_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"Ensemble Model (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="black", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    save_plot("roc_curve.png")

    importance_paths = {
        "dt": MODEL_OUTPUT_DIRS["dt"] / "feature_importance_dt.csv",
        "lr": MODEL_OUTPUT_DIRS["lr"] / "feature_importance_lr.csv",
        "gbm": MODEL_OUTPUT_DIRS["gbm"] / "feature_importance_gbm_optuna.csv",
    }
    if not importance_paths["gbm"].exists():
        importance_paths["gbm"] = MODEL_OUTPUT_DIRS["gbm"] / "feature_importance_gbm_basic.csv"

    model_importances: dict[str, pd.Series] = {}
    for model_name, path in importance_paths.items():
        if not path.exists():
            logger.warning("Missing feature importance file: %s", path)
            continue
        series = load_feature_importance(path)
        if series.sum() > 0:
            model_importances[model_name] = series / series.sum()

    if model_importances:
        available_models = [m for m in best_weights if m in model_importances]
        total_weight = sum(best_weights[m] for m in available_models)
        if total_weight <= 0:
            weights = {m: 1.0 / len(available_models) for m in available_models}
        else:
            weights = {m: best_weights[m] / total_weight for m in available_models}

        all_features = sorted({f for s in model_importances.values() for f in s.index})
        combined = pd.Series(0.0, index=all_features)
        for model_name, series in model_importances.items():
            if model_name not in weights:
                continue
            combined += weights[model_name] * series.reindex(all_features, fill_value=0)

        imp_df = pd.DataFrame({"feature": combined.index, "importance": combined.values})
        imp_df = imp_df.sort_values(by="importance", ascending=False)
        if imp_df["importance"].sum() > 0:
            imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()
        imp_path = ENSEMBLE_DIR / "feature_importance_ensemble.csv"
        imp_df.to_csv(imp_path, index=False)
        logger.info("Saved aggregated feature importance: %s", imp_path)

        top_n = imp_df.head(20)
        plt.figure(figsize=(10, 8))
        plt.title("Aggregated Feature Importance (Top 20)")
        plt.barh(range(len(top_n)), top_n["importance"], align="center", color="teal")
        plt.yticks(range(len(top_n)), top_n["feature"])
        plt.gca().invert_yaxis()
        save_plot("feature_importance_ensemble.png")
    else:
        logger.warning("No feature importance files found; skipping aggregation")

    logger.info("Loading test predictions")
    test_files = {
        "dt": "test_prediction_dt.csv",
        "lr": "test_prediction_lr.csv",
        "gbm": "test_prediction_gbm.csv",
    }

    dfs_test: dict[str, pd.DataFrame] = {}
    for model_name, file_name in test_files.items():
        path = MODEL_OUTPUT_DIRS[model_name] / file_name
        dfs_test[model_name] = load_test_predictions(path, model_name)

    df_ensemble_test = dfs_test["dt"]
    for model in ["lr", "gbm"]:
        df_ensemble_test = df_ensemble_test.merge(dfs_test[model], on="id", how="inner")

    df_ensemble_test["ensemble_prob"] = (
        df_ensemble_test["prob_dt"] * best_weights["dt"]
        + df_ensemble_test["prob_lr"] * best_weights["lr"]
        + df_ensemble_test["prob_gbm"] * best_weights["gbm"]
    )

    df_ensemble_test["prediction"] = (df_ensemble_test["ensemble_prob"] >= best_threshold).astype(int)
    df_ensemble_test["prediction_label"] = df_ensemble_test["prediction"].map({1: "yes", 0: "no"})

    submission_df = pd.DataFrame(
        {
            "prediction": df_ensemble_test["prediction_label"],
            "score": df_ensemble_test["ensemble_prob"],
        }
    )

    save_path = ENSEMBLE_DIR / "final_ensemble_submission_robust.csv"
    submission_df.to_csv(save_path, index=False)
    logger.info("Saved submission: %s", save_path)
    logger.info("Submission preview:\n%s", submission_df.head().to_string(index=False))

    submit_path = ROOT_DIR / "data" / "submit_2025.csv"
    submit_df = read_csv_with_fallback(submit_path)
    target_col = None
    prob_col = None
    for candidate in ["ｙ", "y", "ВЩ"]:
        if candidate in submit_df.columns:
            target_col = candidate
            break
    if "probability of yes (or score)" in submit_df.columns:
        prob_col = "probability of yes (or score)"
    if target_col is None:
        target_col = submit_df.columns[-2]
    if prob_col is None:
        prob_col = submit_df.columns[-1]

    submit_df[target_col] = df_ensemble_test["prediction_label"].values
    submit_df[prob_col] = df_ensemble_test["ensemble_prob"].values

    full_save_path = ENSEMBLE_DIR / "submit_prediction_ensemble.csv"
    submit_df.to_csv(full_save_path, index=False, encoding="shift_jis")
    logger.info("Saved full submission: %s", full_save_path)


if __name__ == "__main__":
    main()
