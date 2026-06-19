import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

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
    parser = argparse.ArgumentParser(
        description="Train GPU-oriented multi-seed blend (CatBoost + XGBoost) and create Kaggle submission"
    )
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--submission",
        default="outputs/submission_blend_v4_gpu.csv",
        help="Output submission CSV",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/metrics_blend_v4_gpu.json",
        help="Output metrics JSON",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Number of CV folds per seed")
    parser.add_argument(
        "--seeds",
        default="42,2024",
        help="Comma-separated CV seeds (example: 42,2024)",
    )
    parser.add_argument(
        "--device-mode",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Device policy. auto=use GPU when available, cpu=force CPU, gpu=fail if GPU not available",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional stratified train-row cap for quick experiments (0 means full data)",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Thread count for XGBoost (CPU path)")
    parser.add_argument("--verbose-every", type=int, default=300, help="Log every N iterations")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> list[int]:
    values: list[int] = []
    for part in seed_text.split(","):
        token = part.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("At least one seed is required")
    return sorted(set(values))


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


def select_training_subset(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or max_rows >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True)

    indices = np.arange(len(X))
    chosen, _ = train_test_split(
        indices,
        train_size=max_rows,
        stratify=y,
        random_state=random_state,
    )
    chosen = np.sort(chosen)
    return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)


def prepare_xgb_matrices(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([X, X_test], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=cat_features, drop_first=False, dtype=np.uint8)
    combined = combined.astype(np.float32)

    X_xgb = combined.iloc[: len(X)].reset_index(drop=True)
    X_test_xgb = combined.iloc[len(X) :].reset_index(drop=True)
    return X_xgb, X_test_xgb


def detect_catboost_gpu() -> tuple[bool, str]:
    try:
        X_small = pd.DataFrame({"cat": ["a", "b", "a", "b"], "num": [0.0, 1.0, 0.1, 0.9]})
        y_small = np.array([0, 1, 0, 1], dtype=np.int8)
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=5,
            task_type="GPU",
            devices="0",
            verbose=False,
        )
        model.fit(X_small, y_small, cat_features=[0])
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def detect_xgboost_gpu() -> tuple[bool, str]:
    try:
        X_small = np.array(
            [[0.0, 0.1], [1.0, 0.8], [0.2, 0.3], [0.9, 0.7], [0.1, 0.2], [0.8, 0.9]],
            dtype=np.float32,
        )
        y_small = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=5,
            tree_method="hist",
            device="cuda",
            random_state=42,
        )
        model.fit(X_small, y_small, verbose=False)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def resolve_devices(device_mode: str) -> dict[str, str]:
    if device_mode == "cpu":
        return {"catboost_task_type": "CPU", "xgb_device": "cpu"}

    cat_gpu_ok, cat_gpu_msg = detect_catboost_gpu()
    xgb_gpu_ok, xgb_gpu_msg = detect_xgboost_gpu()

    if device_mode == "gpu":
        if not cat_gpu_ok or not xgb_gpu_ok:
            raise RuntimeError(
                "GPU mode requested but GPU checks failed. "
                f"catboost_ok={cat_gpu_ok} ({cat_gpu_msg[:200]}), "
                f"xgboost_ok={xgb_gpu_ok} ({xgb_gpu_msg[:200]})"
            )
        return {"catboost_task_type": "GPU", "xgb_device": "cuda"}

    cat_mode = "GPU" if cat_gpu_ok else "CPU"
    xgb_mode = "cuda" if xgb_gpu_ok else "cpu"
    return {"catboost_task_type": cat_mode, "xgb_device": xgb_mode}


def train_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    task_type: str,
    verbose_every: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_scores: list[float] = []

    cat_indices = [X.columns.get_loc(col) for col in cat_features]

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model_seed = seed * 100 + fold
        params: dict[str, object] = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 7.0,
            "early_stopping_rounds": 250,
            "random_seed": model_seed,
            "task_type": task_type,
            "verbose": verbose_every,
            "train_dir": f"catboost_info/v4_seed_{seed}_fold_{fold}",
        }
        if task_type == "GPU":
            params["devices"] = "0"

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
            raise RuntimeError(
                f"CatBoost failed on {task_type} device at seed={seed}, fold={fold}: {exc}"
            ) from exc

        oof_fold = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = oof_fold
        test_pred += model.predict_proba(X_test)[:, 1] / len(folds)

        fold_auc = float(roc_auc_score(y_va, oof_fold))
        fold_scores.append(fold_auc)
        print(f"[Seed {seed}] CatBoost({task_type}) fold {fold} AUC: {fold_auc:.6f}")

    return oof, test_pred, fold_scores


def train_xgb_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    device: str,
    n_jobs: int,
    verbose_every: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model_seed = seed * 100 + fold
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            device=device,
            random_state=model_seed,
            n_jobs=n_jobs,
            early_stopping_rounds=300,
        )

        try:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=verbose_every,
            )
        except XGBoostError as exc:
            raise RuntimeError(
                f"XGBoost failed on {device} device at seed={seed}, fold={fold}: {exc}"
            ) from exc

        oof_fold = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = oof_fold
        test_pred += model.predict_proba(X_test)[:, 1] / len(folds)

        fold_auc = float(roc_auc_score(y_va, oof_fold))
        fold_scores.append(fold_auc)
        print(f"[Seed {seed}] XGBoost({device}) fold {fold} AUC: {fold_auc:.6f}")

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
    seeds = parse_seed_list(args.seeds)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    all_features, cat_features, _ = build_feature_lists(train_df)
    y_full = map_target(train_df[TARGET_COLUMN])
    X_full = train_df[all_features]
    X_test = test_df[all_features]

    X, y = select_training_subset(X_full, y_full, args.max_train_rows, seeds[0])
    if len(X) != len(X_full):
        print(f"Using stratified subset: {len(X)} rows out of {len(X_full)}")

    X_xgb, X_test_xgb = prepare_xgb_matrices(X, X_test, cat_features)

    device_config = resolve_devices(args.device_mode)
    catboost_task_type = device_config["catboost_task_type"]
    xgb_device = device_config["xgb_device"]
    print(f"Resolved devices: CatBoost={catboost_task_type}, XGBoost={xgb_device}")

    cat_oof_all: list[np.ndarray] = []
    xgb_oof_all: list[np.ndarray] = []
    cat_test_all: list[np.ndarray] = []
    xgb_test_all: list[np.ndarray] = []
    blend_oof_seed_all: list[np.ndarray] = []
    blend_test_seed_all: list[np.ndarray] = []
    per_seed_metrics: list[dict[str, object]] = []

    for seed in seeds:
        print(f"\n===== Running seed {seed} with {args.n_splits}-fold CV =====")
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(X, y))

        cat_oof, cat_test, cat_fold_scores = train_catboost_cv(
            X=X,
            y=y,
            X_test=X_test,
            cat_features=cat_features,
            folds=folds,
            seed=seed,
            task_type=catboost_task_type,
            verbose_every=args.verbose_every,
        )

        xgb_oof, xgb_test, xgb_fold_scores = train_xgb_cv(
            X=X_xgb,
            y=y,
            X_test=X_test_xgb,
            folds=folds,
            seed=seed,
            device=xgb_device,
            n_jobs=args.n_jobs,
            verbose_every=args.verbose_every,
        )

        cat_auc = float(roc_auc_score(y, cat_oof))
        xgb_auc = float(roc_auc_score(y, xgb_oof))
        seed_blend_weight, seed_blend_auc = optimize_blend_weight(y, cat_oof, xgb_oof)
        seed_blend_test = seed_blend_weight * cat_test + (1.0 - seed_blend_weight) * xgb_test
        seed_blend_oof = seed_blend_weight * cat_oof + (1.0 - seed_blend_weight) * xgb_oof

        cat_oof_all.append(cat_oof)
        xgb_oof_all.append(xgb_oof)
        cat_test_all.append(cat_test)
        xgb_test_all.append(xgb_test)
        blend_oof_seed_all.append(seed_blend_oof)
        blend_test_seed_all.append(seed_blend_test)

        print(
            f"[Seed {seed}] CatBoost OOF AUC: {cat_auc:.6f} | "
            f"XGBoost OOF AUC: {xgb_auc:.6f} | "
            f"Seed blend AUC: {seed_blend_auc:.6f} (cat weight={seed_blend_weight:.2f})"
        )

        per_seed_metrics.append(
            {
                "seed": seed,
                "catboost_fold_auc": cat_fold_scores,
                "xgboost_fold_auc": xgb_fold_scores,
                "catboost_oof_auc": cat_auc,
                "xgboost_oof_auc": xgb_auc,
                "seed_blend_weight_catboost": seed_blend_weight,
                "seed_blend_weight_xgboost": 1.0 - seed_blend_weight,
                "seed_blend_oof_auc": seed_blend_auc,
            }
        )

    cat_oof_mean = np.mean(np.vstack(cat_oof_all), axis=0)
    xgb_oof_mean = np.mean(np.vstack(xgb_oof_all), axis=0)
    cat_test_mean = np.mean(np.vstack(cat_test_all), axis=0)
    xgb_test_mean = np.mean(np.vstack(xgb_test_all), axis=0)

    cat_oof_auc_mean = float(roc_auc_score(y, cat_oof_mean))
    xgb_oof_auc_mean = float(roc_auc_score(y, xgb_oof_mean))

    model_avg_weight_cat, model_avg_blend_auc = optimize_blend_weight(y, cat_oof_mean, xgb_oof_mean)
    model_avg_test_blend = model_avg_weight_cat * cat_test_mean + (1.0 - model_avg_weight_cat) * xgb_test_mean

    seed_avg_oof_blend = np.mean(np.vstack(blend_oof_seed_all), axis=0)
    seed_avg_test_blend = np.mean(np.vstack(blend_test_seed_all), axis=0)
    seed_avg_blend_auc = float(roc_auc_score(y, seed_avg_oof_blend))

    if model_avg_blend_auc >= seed_avg_blend_auc:
        chosen_strategy = "model_average_then_optimize"
        final_pred = model_avg_test_blend
        final_oof_auc = model_avg_blend_auc
        final_weight_cat = model_avg_weight_cat
    else:
        chosen_strategy = "average_seed_blends"
        final_pred = seed_avg_test_blend
        final_oof_auc = seed_avg_blend_auc
        final_weight_cat = None

    submission_df = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: final_pred})
    validate_submission(submission_df, len(test_df))

    submission_path = Path(args.submission)
    metrics_path = Path(args.metrics)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

    metrics = {
        "cv_splits": args.n_splits,
        "seeds": seeds,
        "device_mode_requested": args.device_mode,
        "device_resolved": {
            "catboost_task_type": catboost_task_type,
            "xgboost_device": xgb_device,
        },
        "max_train_rows": args.max_train_rows,
        "used_train_rows": int(len(X)),
        "xgboost_feature_count": int(X_xgb.shape[1]),
        "per_seed": per_seed_metrics,
        "catboost_oof_auc_seed_mean": cat_oof_auc_mean,
        "xgboost_oof_auc_seed_mean": xgb_oof_auc_mean,
        "model_avg_then_optimize": {
            "catboost_weight": model_avg_weight_cat,
            "xgboost_weight": 1.0 - model_avg_weight_cat,
            "oof_auc": model_avg_blend_auc,
        },
        "average_seed_blends": {
            "oof_auc": seed_avg_blend_auc,
        },
        "chosen_strategy": chosen_strategy,
        "chosen_oof_auc": final_oof_auc,
        "chosen_catboost_weight_if_applicable": final_weight_cat,
        "train_rows_total": int(len(X_full)),
        "test_rows": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n===== V4 summary =====")
    print(f"CatBoost mean-seed OOF AUC: {cat_oof_auc_mean:.6f}")
    print(f"XGBoost mean-seed OOF AUC: {xgb_oof_auc_mean:.6f}")
    print(f"Model-average-then-optimize OOF AUC: {model_avg_blend_auc:.6f}")
    print(f"Average-seed-blends OOF AUC: {seed_avg_blend_auc:.6f}")
    print(f"Chosen strategy: {chosen_strategy}")
    print(f"Chosen OOF AUC: {final_oof_auc:.6f}")
    print(f"Submission written to: {submission_path}")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
