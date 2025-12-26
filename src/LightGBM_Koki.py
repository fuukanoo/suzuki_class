from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("LightGBM is required to run this script.") from exc

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output" / "LightGBM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_OPTUNA = True
RUN_SHAP = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "lightgbm.log", encoding="utf-8"),
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


def save_feature_importance(model, filename: str) -> None:
    importance = model.feature_importance(importance_type="gain")
    features = model.feature_name()
    imp_df = pd.DataFrame({"feature": features, "importance": importance})
    imp_df = imp_df[imp_df["importance"] > 0].copy()
    if not imp_df.empty:
        imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()
    imp_df = imp_df.sort_values(by="importance", ascending=False)
    output_path = OUTPUT_DIR / filename
    imp_df.to_csv(output_path, index=False)
    logger.info("Saved aggregated feature importance: %s", output_path)


def preprocess_training(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["id"] = df.index

    cols_to_drop = ["education", "fnlwgt"]
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    if existing_drop_cols:
        df = df.drop(existing_drop_cols, axis=1)
        logger.info("Dropped columns: %s", existing_drop_cols)

    target_col = resolve_target_column(df)
    df = df.rename(columns={target_col: "target"})
    df["target"] = df["target"].astype(str).str.strip().map({"yes": 1, "no": 0})

    if "capital-gain" in df.columns:
        df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
        df["capital-gain-log"] = np.log1p(df["capital-gain"])
    if "capital-loss" in df.columns:
        df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)
        df["capital-loss-log"] = np.log1p(df["capital-loss"])

    drop_capital = [c for c in ["capital-gain", "capital-loss"] if c in df.columns]
    if drop_capital:
        df = df.drop(drop_capital, axis=1)

    cat_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    actual_cat_cols = [c for c in cat_cols if c in df.columns]
    for col in actual_cat_cols:
        df[col] = df[col].astype("category")

    return df, actual_cat_cols


def preprocess_submit(
    df_submit: pd.DataFrame, feature_columns: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    df_submit = df_submit.copy()
    cols_to_drop = ["education", "fnlwgt", "ｙ", "probability of yes (or score)"]
    df_submit = df_submit.drop([c for c in cols_to_drop if c in df_submit.columns], axis=1)

    if "capital-gain" in df_submit.columns:
        df_submit["has_capital_gain"] = (df_submit["capital-gain"] > 0).astype(int)
        df_submit["capital-gain-log"] = np.log1p(df_submit["capital-gain"])
    if "capital-loss" in df_submit.columns:
        df_submit["has_capital_loss"] = (df_submit["capital-loss"] > 0).astype(int)
        df_submit["capital-loss-log"] = np.log1p(df_submit["capital-loss"])

    drop_capital = [c for c in ["capital-gain", "capital-loss"] if c in df_submit.columns]
    if drop_capital:
        df_submit = df_submit.drop(drop_capital, axis=1)

    for col in cat_cols:
        if col in df_submit.columns:
            df_submit[col] = df_submit[col].astype("category")

    df_submit = df_submit.reindex(columns=feature_columns, fill_value=0)
    return df_submit


def write_outputs(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    id_val: pd.Series,
    df_submit: pd.DataFrame,
    feature_columns: list[str],
    cat_cols: list[str],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_val_prob = model.predict(X_val)
    df_valid_output = pd.DataFrame(
        {
            "id": id_val,
            "Actual": y_val,
            "GBM_Probability": y_val_prob,
            "GBM_Prediction": (y_val_prob >= threshold).astype(int),
        }
    )
    valid_output_path = OUTPUT_DIR / "valid_prediction_gbm.csv"
    df_valid_output.to_csv(valid_output_path, index=False)
    logger.info("Saved validation predictions: %s", valid_output_path)

    df_submit_proc = preprocess_submit(df_submit, feature_columns, cat_cols)
    submit_probs = model.predict(df_submit_proc)

    df_submit_output = pd.DataFrame(
        {
            "id": df_submit["id"],
            "GBM_Probability": submit_probs,
            "GBM_Prediction": (submit_probs >= threshold).astype(int),
        }
    )
    submit_output_path = OUTPUT_DIR / "test_prediction_gbm.csv"
    df_submit_output.to_csv(submit_output_path, index=False)
    logger.info("Saved test predictions: %s", submit_output_path)

    output_file_path = OUTPUT_DIR / "submit_prediction_lightgbm.csv"
    df_final_submit = df_submit.copy()
    df_final_submit["probability of yes (or score)"] = submit_probs
    pred_binary = (submit_probs >= threshold).astype(int)
    df_final_submit["ｙ"] = pred_binary
    df_final_submit["ｙ"] = df_final_submit["ｙ"].map({1: "yes", 0: "no"})
    df_final_submit = df_final_submit.drop(columns=["id"], errors="ignore")
    df_final_submit.to_csv(output_file_path, index=False, encoding="shift_jis")
    logger.info("Saved submission file: %s", output_file_path)

    return y_val_prob, submit_probs


def train_basic_model(X_train, y_train, X_val, y_val, cat_cols):
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "is_unbalance": True,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
    }

    logger.info("LightGBM training start")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100),
        ],
    )

    y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
    auc_score = roc_auc_score(y_val, y_pred_prob)
    logger.info("Validation AUC: %.4f", auc_score)

    plt.figure(figsize=(10, 8))
    lgb.plot_importance(
        model,
        importance_type="gain",
        max_num_features=20,
        title="Feature Importance (Gain)",
        figsize=(10, 8),
    )
    save_plot("feature_importance_basic.png")

    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"LightGBM (AUC = {auc_score:.4f})", color="blue", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    save_plot("roc_curve_basic.png")

    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
    )
    plt.title("Confusion Matrix (Threshold=0.5)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    save_plot("confusion_matrix_basic.png")

    save_feature_importance(model, "feature_importance_gbm_basic.csv")
    logger.info("Classification report:\n%s", classification_report(y_val, y_pred_binary))
    return model


def run_optuna_training(
    df: pd.DataFrame,
    cat_cols: list[str],
    df_submit: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna is not installed; skipping hyperparameter search")
        return

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

    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "is_unbalance": True,
            "random_state": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        for train_idx, valid_idx in skf.split(X_train, y_train):
            X_tr_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            train_data = lgb.Dataset(X_tr_cv, label=y_tr_cv, categorical_feature=cat_cols)
            val_data = lgb.Dataset(
                X_val_cv, label=y_val_cv, categorical_feature=cat_cols, reference=train_data
            )
            model = lgb.train(
                param,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            preds = model.predict(X_val_cv, num_iteration=model.best_iteration)
            auc_scores.append(roc_auc_score(y_val_cv, preds))
        return float(np.mean(auc_scores))

    logger.info("Optuna study start")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    best_params.update(
        {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "is_unbalance": True,
            "random_state": 42,
        }
    )

    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    final_model = lgb.train(best_params, lgb_train, num_boost_round=1000)

    y_val_prob = final_model.predict(X_val)

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_threshold = 0.0, 0.5
    for th in thresholds:
        f1 = f1_score(y_val, (y_val_prob >= th).astype(int))
        if f1 > best_f1:
            best_f1, best_threshold = f1, th

    y_val_prob, submit_probs = write_outputs(
        final_model,
        X_val,
        y_val,
        id_val,
        df_submit,
        feature_columns,
        cat_cols,
        best_threshold,
    )

    auc_score = roc_auc_score(y_val, y_val_prob)
    logger.info("Validation AUC: %.4f", auc_score)

    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"LightGBM (AUC = {auc_score:.4f})", color="blue", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    save_plot("roc_curve_optuna.png")

    plt.figure(figsize=(10, 8))
    lgb.plot_importance(
        final_model,
        importance_type="gain",
        max_num_features=20,
        title="Feature Importance (Gain)",
        figsize=(10, 8),
    )
    save_plot("feature_importance_optuna.png")
    save_feature_importance(final_model, "feature_importance_gbm_optuna.csv")

    y_val_pred = (y_val_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_val, y_val_pred)
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
    save_plot("confusion_matrix_optuna.png")

    logger.info("Classification report:\n%s", classification_report(y_val, y_val_pred))

    if not RUN_SHAP:
        logger.info("SHAP is disabled; set RUN_SHAP=True to enable")
        return

    try:
        import shap
    except ImportError:
        logger.warning("SHAP is not installed; skipping SHAP analysis")
        return

    X_test = preprocess_submit(df_submit, feature_columns, cat_cols)
    if X_test.empty:
        X_test = X_val

    sample_size = min(1000, len(X_test))
    shap_sample = X_test.sample(n=sample_size, random_state=42)

    logger.info("SHAP analysis start")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        shap_values_target = shap_values[1]
    else:
        shap_values_target = shap_values

    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values_target, shap_sample, show=False)
    plt.title("SHAP Summary Plot")
    save_plot("shap_summary.png")

    target_feature = "age"
    if target_feature in shap_sample.columns:
        shap.dependence_plot(target_feature, shap_values_target, shap_sample, show=False)
        plt.title(f"SHAP Dependence Plot: {target_feature}")
        save_plot("shap_dependence_age.png")


def main() -> None:
    train_path = DATA_DIR / "train_test_2025.csv"
    df = read_csv_with_fallback(train_path)
    df, cat_cols = preprocess_training(df)

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

    df_submit = read_csv_with_fallback(DATA_DIR / "submit_2025.csv")
    df_submit["id"] = df_submit.index

    base_model = train_basic_model(X_train, y_train, X_val, y_val, cat_cols)
    base_val_prob = base_model.predict(X_val, num_iteration=base_model.best_iteration)

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_threshold = 0.0, 0.5
    for th in thresholds:
        f1 = f1_score(y_val, (base_val_prob >= th).astype(int))
        if f1 > best_f1:
            best_f1, best_threshold = f1, th

    logger.info("Baseline threshold: %.2f (F1=%.4f)", best_threshold, best_f1)
    write_outputs(
        base_model,
        X_val,
        y_val,
        id_val,
        df_submit,
        X_train.columns.tolist(),
        cat_cols,
        best_threshold,
    )

    if RUN_OPTUNA:
        run_optuna_training(df, cat_cols, df_submit, X_train.columns.tolist())
    else:
        logger.info("Optuna is disabled; set RUN_OPTUNA=True to enable")


if __name__ == "__main__":
    main()
