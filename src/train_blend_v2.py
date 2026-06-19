import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET_COLUMN = "Churn"
ID_COLUMN = "id"
BASE_CAT_FEATURES = [
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
BASE_NUM_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CV blend (CatBoost + LightGBM) and create Kaggle submission")
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument("--submission", default="outputs/submission_blend_v2.csv", help="Output submission CSV")
    parser.add_argument("--metrics", default="outputs/metrics_blend_v2.json", help="Output metrics JSON")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="CV random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Thread count for LightGBM")
    parser.add_argument(
        "--device-mode",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="CatBoost device mode. auto=try GPU then CPU fallback",
    )
    parser.add_argument("--gpu-devices", default="0", help="CatBoost GPU devices value when using GPU mode")
    return parser.parse_args()


def map_target(target: pd.Series) -> pd.Series:
    mapped = target.map({"No": 0, "Yes": 1})
    if mapped.isna().any():
        bad = sorted(target[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected target values in {TARGET_COLUMN}: {bad}")
    return mapped.astype("int8")


def _is_yes(series: pd.Series) -> pd.Series:
    return (series == "Yes").astype("int8")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["charge_per_tenure"] = data["TotalCharges"] / (data["tenure"] + 1.0)
    data["monthly_to_total_ratio"] = data["MonthlyCharges"] / (data["TotalCharges"] + 1.0)
    data["charges_gap"] = data["TotalCharges"] - (data["MonthlyCharges"] * data["tenure"])
    data["charges_gap_abs"] = data["charges_gap"].abs()

    data["is_new_customer"] = (data["tenure"] <= 12).astype("int8")
    data["is_long_customer"] = (data["tenure"] >= 48).astype("int8")
    data["is_month_to_month"] = (data["Contract"] == "Month-to-month").astype("int8")
    data["is_electronic_check"] = (data["PaymentMethod"] == "Electronic check").astype("int8")
    data["is_autopay"] = data["PaymentMethod"].astype(str).str.contains("(automatic)", regex=False).astype("int8")

    internet_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    data["internet_services_yes"] = sum(_is_yes(data[col]) for col in internet_cols).astype("int8")

    data["has_internet"] = (data["InternetService"] != "No").astype("int8")
    data["has_phone"] = (data["PhoneService"] == "Yes").astype("int8")

    data["contract_term_ordinal"] = (
        data["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}).fillna(-1).astype("int8")
    )
    data["tenure_bin"] = pd.cut(
        data["tenure"],
        bins=[-1, 6, 12, 24, 36, 48, 60, 72, 10**9],
        labels=["0_6", "7_12", "13_24", "25_36", "37_48", "49_60", "61_72", "73_plus"],
    ).astype(str)
    data["monthly_charge_bin"] = pd.cut(
        data["MonthlyCharges"],
        bins=[-1, 35, 70, 90, 10**9],
        labels=["low_lt35", "mid_35_70", "high_70_90", "very_high_ge90"],
    ).astype(str)
    data["internet_services_yes_bucket"] = pd.cut(
        data["internet_services_yes"],
        bins=[-1, 0, 2, 4, 6],
        labels=["0", "1_2", "3_4", "5_6"],
    ).astype(str)

    data["contract_payment_combo"] = data["Contract"].astype(str) + "__" + data["PaymentMethod"].astype(str)
    data["service_profile"] = data["InternetService"].astype(str) + "__" + data["MultipleLines"].astype(str)
    data["payment_paperless_combo"] = data["PaymentMethod"].astype(str) + "__" + data["PaperlessBilling"].astype(str)
    data["monthly_charge_bin_contract"] = data["monthly_charge_bin"] + "__" + data["Contract"].astype(str)
    data["household_stability"] = (
        data["Partner"].astype(str) + "__" + data["Dependents"].astype(str) + "__" + data["Contract"].astype(str)
    )

    return data


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    extra_cat = [
        "contract_payment_combo",
        "service_profile",
        "payment_paperless_combo",
        "tenure_bin",
        "monthly_charge_bin",
        "internet_services_yes_bucket",
        "monthly_charge_bin_contract",
        "household_stability",
    ]
    extra_num = [
        "charge_per_tenure",
        "monthly_to_total_ratio",
        "charges_gap",
        "charges_gap_abs",
        "is_new_customer",
        "is_long_customer",
        "is_month_to_month",
        "is_electronic_check",
        "is_autopay",
        "contract_term_ordinal",
        "internet_services_yes",
        "has_internet",
        "has_phone",
    ]

    cat_features = BASE_CAT_FEATURES + extra_cat
    num_features = BASE_NUM_FEATURES + extra_num
    all_features = cat_features + num_features

    missing = [col for col in all_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required engineered features: {missing}")

    return all_features, cat_features, num_features


def detect_catboost_gpu(gpu_devices: str) -> tuple[bool, str]:
    X_small = pd.DataFrame(
        {
            "cat_col": ["a", "b", "a", "c", "b", "c"],
            "num_col": [0.1, 0.2, 0.3, 0.5, 0.4, 0.6],
        }
    )
    y_small = pd.Series([0, 1, 0, 1, 0, 1], dtype="int8")
    try:
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=5,
            learning_rate=0.1,
            task_type="GPU",
            devices=gpu_devices,
            verbose=False,
        )
        model.fit(X_small, y_small, cat_features=[0])
        return True, "gpu_ok"
    except Exception as exc:
        return False, str(exc)


def resolve_catboost_task_type(device_mode: str, gpu_devices: str) -> tuple[str, str]:
    if device_mode == "cpu":
        return "CPU", "forced_cpu"
    gpu_ok, reason = detect_catboost_gpu(gpu_devices)
    if device_mode == "gpu":
        if not gpu_ok:
            raise RuntimeError(f"Requested GPU mode but CatBoost GPU is unavailable: {reason}")
        return "GPU", "forced_gpu"
    if gpu_ok:
        return "GPU", "auto_gpu"
    return "CPU", f"auto_cpu_fallback: {reason}"


def train_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
    skf: StratifiedKFold,
    task_type: str,
    gpu_devices: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_scores: list[float] = []

    cat_indices = [X.columns.get_loc(col) for col in cat_features]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        params: dict[str, object] = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 7.0,
            "subsample": 0.85,
            "random_strength": 0.2,
            "early_stopping_rounds": 250,
            "random_seed": 42 + fold,
            "task_type": task_type,
            "verbose": 300,
            "train_dir": f"catboost_info/blend_fold_{fold}",
        }
        if task_type == "GPU":
            params["devices"] = gpu_devices
            params.pop("subsample", None)

        model = CatBoostClassifier(**params)

        try:
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_indices,
                eval_set=(X_va, y_va),
                use_best_model=True,
            )
        except CatBoostError as exc:
            raise RuntimeError(f"CatBoost failed on {task_type} at fold {fold}: {exc}") from exc

        oof_fold = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = oof_fold
        test_pred += model.predict_proba(X_test)[:, 1] / skf.get_n_splits()

        fold_auc = float(roc_auc_score(y_va, oof_fold))
        fold_scores.append(fold_auc)
        print(f"CatBoost({task_type}) fold {fold} AUC: {fold_auc:.6f}")

    return oof, test_pred, fold_scores


def train_lgbm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
    skf: StratifiedKFold,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_scores: list[float] = []

    X_lgb = X.copy()
    X_test_lgb = X_test.copy()
    for col in cat_features:
        X_lgb[col] = X_lgb[col].astype("category")
        X_test_lgb[col] = X_test_lgb[col].astype("category")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_lgb, y), start=1):
        X_tr = X_lgb.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X_lgb.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model = LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=120,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.5,
            n_jobs=n_jobs,
            random_state=100 + fold,
            verbosity=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[early_stopping(300), log_evaluation(300)],
        )

        oof_fold = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = oof_fold
        test_pred += model.predict_proba(X_test_lgb)[:, 1] / skf.get_n_splits()

        fold_auc = float(roc_auc_score(y_va, oof_fold))
        fold_scores.append(fold_auc)
        print(f"LightGBM fold {fold} AUC: {fold_auc:.6f}")

    return oof, test_pred, fold_scores


def optimize_blend_weight(y: pd.Series, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    best_weight = 0.5
    best_auc = -1.0

    for weight in np.linspace(0.0, 1.0, 101):
        blend = weight * pred_a + (1.0 - weight) * pred_b
        auc = float(roc_auc_score(y, blend))
        if auc > best_auc:
            best_auc = auc
            best_weight = float(weight)

    return best_weight, best_auc


def validate_submission(df: pd.DataFrame, expected_rows: int) -> None:
    if list(df.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("Submission columns must be exactly: id,Churn")
    if len(df) != expected_rows:
        raise ValueError(f"Submission row count mismatch. Expected {expected_rows}, got {len(df)}")
    probs = df[TARGET_COLUMN].to_numpy()
    if np.isnan(probs).any() or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Submission probabilities must be in [0, 1] and non-NaN")


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    all_features, cat_features, _ = build_feature_lists(train_df)

    y = map_target(train_df[TARGET_COLUMN])
    X = train_df[all_features]
    X_test = test_df[all_features]

    cat_task_type, cat_device_note = resolve_catboost_task_type(args.device_mode, args.gpu_devices)
    print(f"CatBoost device resolved: {cat_task_type} ({cat_device_note})")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    cat_oof, cat_test, cat_scores = train_catboost_cv(
        X, y, X_test, cat_features, skf, cat_task_type, args.gpu_devices
    )
    lgb_oof, lgb_test, lgb_scores = train_lgbm_cv(X, y, X_test, cat_features, skf, args.n_jobs)

    cat_auc = float(roc_auc_score(y, cat_oof))
    lgb_auc = float(roc_auc_score(y, lgb_oof))

    blend_weight_cat, blend_auc = optimize_blend_weight(y, cat_oof, lgb_oof)
    blend_weight_lgb = 1.0 - blend_weight_cat
    test_blend = blend_weight_cat * cat_test + blend_weight_lgb * lgb_test

    submission_df = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: test_blend})
    validate_submission(submission_df, len(test_df))

    submission_path = Path(args.submission)
    metrics_path = Path(args.metrics)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df.to_csv(submission_path, index=False)

    metrics = {
        "cv_splits": args.n_splits,
        "device_mode_requested": args.device_mode,
        "catboost_task_type_resolved": cat_task_type,
        "catboost_device_note": cat_device_note,
        "catboost_fold_auc": cat_scores,
        "lightgbm_fold_auc": lgb_scores,
        "catboost_oof_auc": cat_auc,
        "lightgbm_oof_auc": lgb_auc,
        "blend_weight_catboost": blend_weight_cat,
        "blend_weight_lightgbm": blend_weight_lgb,
        "blend_oof_auc": blend_auc,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"CatBoost OOF AUC: {cat_auc:.6f}")
    print(f"LightGBM OOF AUC: {lgb_auc:.6f}")
    print(f"Blend OOF AUC: {blend_auc:.6f} (catboost weight={blend_weight_cat:.2f})")
    print(f"Submission written to: {submission_path}")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
