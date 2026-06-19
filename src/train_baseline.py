import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Churn"
ID_COLUMN = "id"
CAT_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUM_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
EXPECTED_TRAIN_COLUMNS = [ID_COLUMN, *CAT_FEATURES, *NUM_FEATURES, TARGET_COLUMN]
EXPECTED_TEST_COLUMNS = [ID_COLUMN, *CAT_FEATURES, *NUM_FEATURES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost baseline and create Kaggle submission")
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument("--submission", default="outputs/submission_baseline.csv", help="Output submission CSV")
    parser.add_argument("--metrics", default="outputs/metrics_baseline.json", help="Output metrics JSON")
    parser.add_argument("--model", default="artifacts/catboost_baseline.cbm", help="Output CatBoost model file")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def map_target(target: pd.Series) -> pd.Series:
    mapped = target.map({"No": 0, "Yes": 1})
    if mapped.isna().any():
        bad = sorted(target[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected target values in {TARGET_COLUMN}: {bad}")
    return mapped.astype("int8")


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        early_stopping_rounds=200,
        random_seed=42,
        task_type="CPU",
        verbose=200,
    )


def ensure_probability_bounds(probs: np.ndarray) -> None:
    if np.isnan(probs).any():
        raise ValueError("Submission probabilities contain NaN values")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Submission probabilities must be within [0, 1]")


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    ensure_columns(train_df, EXPECTED_TRAIN_COLUMNS, "train")
    ensure_columns(test_df, EXPECTED_TEST_COLUMNS, "test")

    X = train_df[CAT_FEATURES + NUM_FEATURES].copy()
    y = map_target(train_df[TARGET_COLUMN])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    cat_feature_indices = [X.columns.get_loc(col) for col in CAT_FEATURES]

    model = build_model()
    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    valid_pred = model.predict_proba(X_valid)[:, 1]
    roc_auc_valid = float(roc_auc_score(y_valid, valid_pred))

    best_iteration = int(model.get_best_iteration())
    if best_iteration <= 0:
        best_iteration = 2000

    final_model = build_model()
    final_model.set_params(iterations=best_iteration)
    final_model.fit(X, y, cat_features=cat_feature_indices)

    test_pred = final_model.predict_proba(test_df[CAT_FEATURES + NUM_FEATURES])[:, 1]
    ensure_probability_bounds(test_pred)

    submission_df = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        TARGET_COLUMN: test_pred,
    })

    if list(submission_df.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("Submission columns must be exactly: id,Churn")

    if len(submission_df) != len(test_df):
        raise ValueError(
            f"Submission row count mismatch. Expected {len(test_df)}, got {len(submission_df)}"
        )

    submission_path = Path(args.submission)
    metrics_path = Path(args.metrics)
    model_path = Path(args.model)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df.to_csv(submission_path, index=False)
    final_model.save_model(str(model_path))

    metrics = {
        "roc_auc_valid": roc_auc_valid,
        "best_iteration": best_iteration,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target_positive_rate": float(y.mean()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Validation ROC-AUC: {roc_auc_valid:.6f}")
    print(f"Best iteration: {best_iteration}")
    print(f"Submission written to: {submission_path}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Model written to: {model_path}")


if __name__ == "__main__":
    main()
