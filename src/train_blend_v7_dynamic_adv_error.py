import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
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
        description=(
            "Train dynamic blend with adversarial reweighting, row-wise dynamic merge, "
            "and error-focused second-stage correction."
        )
    )
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--submission",
        default="outputs/submission_blend_v7_dynamic_adv_error.csv",
        help="Output submission CSV",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/metrics_blend_v7_dynamic_adv_error.json",
        help="Output metrics JSON",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU thread count")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional stratified row cap")
    parser.add_argument("--device-mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--gpu-devices", default="0", help="CatBoost GPU device string")

    parser.add_argument("--te-cv", type=int, default=5, help="TargetEncoder CV folds")
    parser.add_argument("--te-smooth", default="auto", help="TargetEncoder smooth")

    parser.add_argument("--cat-iterations", type=int, default=2800)
    parser.add_argument("--cat-lr", type=float, default=0.03)
    parser.add_argument("--cat-depth", type=int, default=8)
    parser.add_argument("--cat-es-rounds", type=int, default=250)

    parser.add_argument("--lgb-estimators", type=int, default=4000)
    parser.add_argument("--lgb-lr", type=float, default=0.02)
    parser.add_argument("--lgb-es-rounds", type=int, default=300)

    parser.add_argument("--xgb-estimators", type=int, default=2200)
    parser.add_argument("--xgb-lr", type=float, default=0.03)
    parser.add_argument("--xgb-depth", type=int, default=6)
    parser.add_argument("--xgb-es-rounds", type=int, default=250)

    parser.add_argument("--adv-n-splits", type=int, default=5)
    parser.add_argument("--adv-estimators", type=int, default=1200)
    parser.add_argument("--adv-lr", type=float, default=0.03)
    parser.add_argument("--adv-weight-min", type=float, default=0.60)
    parser.add_argument("--adv-weight-max", type=float, default=1.70)

    parser.add_argument("--dynamic-base-weight", type=float, default=0.05)
    parser.add_argument("--dynamic-uncertainty-weight", type=float, default=0.12)
    parser.add_argument("--dynamic-disagreement-weight", type=float, default=0.10)
    parser.add_argument("--dynamic-max-weight", type=float, default=0.35)

    parser.add_argument("--error-quantile", type=float, default=0.75)
    parser.add_argument("--error-corr-scale", type=float, default=0.10)
    parser.add_argument("--error-estimators", type=int, default=800)
    parser.add_argument("--error-lr", type=float, default=0.02)
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


def select_training_subset(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or max_rows >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True)
    indices = np.arange(len(X))
    chosen, _ = train_test_split(indices, train_size=max_rows, stratify=y, random_state=random_state)
    chosen = np.sort(chosen)
    return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)


def detect_catboost_gpu(gpu_devices: str) -> tuple[bool, str]:
    X_small = pd.DataFrame({"cat_col": ["a", "b", "a", "c"], "num_col": [0.1, 0.2, 0.3, 0.4]})
    y_small = pd.Series([0, 1, 0, 1], dtype="int8")
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


def detect_xgb_gpu() -> tuple[bool, str]:
    X_small = np.array([[0.0, 0.1], [1.0, 0.3], [0.0, 0.2], [1.0, 0.4]], dtype=np.float32)
    y_small = np.array([0, 1, 0, 1], dtype=np.int32)
    try:
        model = XGBClassifier(
            n_estimators=8,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_small, y_small, verbose=False)
        return True, "gpu_ok"
    except Exception as exc:
        return False, str(exc)


def resolve_devices(device_mode: str, gpu_devices: str) -> tuple[str, str, str]:
    if device_mode == "cpu":
        return "CPU", "cpu", "forced_cpu"

    cat_ok, cat_reason = detect_catboost_gpu(gpu_devices)
    xgb_ok, xgb_reason = detect_xgb_gpu()

    if device_mode == "gpu":
        if not cat_ok:
            raise RuntimeError(f"Requested GPU mode but CatBoost GPU unavailable: {cat_reason}")
        if not xgb_ok:
            raise RuntimeError(f"Requested GPU mode but XGBoost GPU unavailable: {xgb_reason}")
        return "GPU", "cuda", "forced_gpu"

    if cat_ok and xgb_ok:
        return "GPU", "cuda", "auto_gpu"
    return "CPU", "cpu", f"auto_cpu_fallback: cat={cat_reason}, xgb={xgb_reason}"


def fit_target_encoder(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list[str],
    te_cv: int,
    te_smooth: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth_value: str | float
    if te_smooth == "auto":
        smooth_value = "auto"
    else:
        smooth_value = float(te_smooth)

    encoder = TargetEncoder(categories="auto", target_type="binary", smooth=smooth_value, cv=te_cv, random_state=seed)
    tr_enc = encoder.fit_transform(X_tr[cat_features], y_tr)
    va_enc = encoder.transform(X_va[cat_features])
    te_enc = encoder.transform(X_test[cat_features])
    return (
        np.asarray(tr_enc, dtype=np.float32),
        np.asarray(va_enc, dtype=np.float32),
        np.asarray(te_enc, dtype=np.float32),
    )


def train_catboost_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
    sample_weight: np.ndarray,
    seed: int,
    cat_task_type: str,
    gpu_devices: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    cat_indices = [X_tr.columns.get_loc(col) for col in cat_features]
    params: dict[str, object] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": args.cat_iterations,
        "learning_rate": args.cat_lr,
        "depth": args.cat_depth,
        "l2_leaf_reg": 7.0,
        "subsample": 0.85,
        "random_strength": 0.2,
        "early_stopping_rounds": args.cat_es_rounds,
        "random_seed": seed,
        "task_type": cat_task_type,
        "verbose": 300,
    }
    if cat_task_type == "GPU":
        params["devices"] = gpu_devices
        params.pop("subsample", None)
    model = CatBoostClassifier(**params)
    try:
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_indices,
            sample_weight=sample_weight,
            eval_set=(X_va, y_va),
            use_best_model=True,
        )
    except CatBoostError as exc:
        raise RuntimeError(f"CatBoost failed on {cat_task_type}: {exc}") from exc
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_test)[:, 1]


def train_lgb_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
    sample_weight: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    Xtr = X_tr.copy()
    Xva = X_va.copy()
    Xte = X_test.copy()
    for col in cat_features:
        Xtr[col] = Xtr[col].astype("category")
        Xva[col] = Xva[col].astype("category")
        Xte[col] = Xte[col].astype("category")

    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=args.lgb_estimators,
        learning_rate=args.lgb_lr,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=120,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.5,
        n_jobs=args.n_jobs,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        Xtr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(Xva, y_va)],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(args.lgb_es_rounds), log_evaluation(300)],
    )
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]


def train_xgb_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_test: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
    xgb_device: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=args.xgb_estimators,
        learning_rate=args.xgb_lr,
        max_depth=args.xgb_depth,
        min_child_weight=4.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        device=xgb_device,
        random_state=seed,
        n_jobs=args.n_jobs,
        early_stopping_rounds=args.xgb_es_rounds,
    )
    try:
        model.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_va, y_va)], verbose=False)
    except XGBoostError as exc:
        raise RuntimeError(f"XGBoost failed on {xgb_device}: {exc}") from exc
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_test)[:, 1]


def optimize_anchor_weight(y: np.ndarray, cat_oof: np.ndarray, lgb_oof: np.ndarray) -> tuple[float, float]:
    best_w = 0.5
    best_auc = -1.0
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * cat_oof + (1.0 - w) * lgb_oof
        auc = float(roc_auc_score(y, blend))
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)
    return best_w, best_auc


def build_dynamic_blend(
    anchor_pred: np.ndarray,
    diversity_pred: np.ndarray,
    base_weight: float,
    uncertainty_weight: float,
    disagreement_weight: float,
    max_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uncertainty = 1.0 - np.clip(np.abs(anchor_pred - 0.5) * 2.0, 0.0, 1.0)
    disagreement = np.abs(anchor_pred - diversity_pred)
    scale = float(np.percentile(disagreement, 95))
    if scale <= 1e-12:
        disagreement_norm = np.zeros_like(disagreement)
    else:
        disagreement_norm = np.clip(disagreement / scale, 0.0, 1.0)
    dynamic_weight = base_weight + uncertainty_weight * uncertainty + disagreement_weight * disagreement_norm
    dynamic_weight = np.clip(dynamic_weight, 0.0, max_weight)
    dynamic_pred = (1.0 - dynamic_weight) * anchor_pred + dynamic_weight * diversity_pred
    return dynamic_pred, dynamic_weight, uncertainty, disagreement_norm


def build_adversarial_features(df: pd.DataFrame, cat_features: list[str], num_features: list[str]) -> np.ndarray:
    work = df[cat_features + num_features].copy()
    for col in cat_features:
        work[col] = work[col].astype("category").cat.codes.astype(np.int32)
    return work.to_numpy(dtype=np.float32)


def compute_adversarial_weights(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list[str],
    num_features: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    X_adv = build_adversarial_features(combined, cat_features, num_features)
    y_adv = np.zeros(len(combined), dtype=np.int8)
    y_adv[len(X_train) :] = 1

    skf_adv = StratifiedKFold(n_splits=args.adv_n_splits, shuffle=True, random_state=args.random_state + 17)
    oof_adv = np.zeros(len(combined), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf_adv.split(X_adv, y_adv), start=1):
        model = LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=args.adv_estimators,
            learning_rate=args.adv_lr,
            num_leaves=64,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.1,
            n_jobs=args.n_jobs,
            random_state=args.random_state * 100 + fold,
            verbosity=-1,
        )
        model.fit(
            X_adv[tr_idx],
            y_adv[tr_idx],
            eval_set=[(X_adv[va_idx], y_adv[va_idx])],
            eval_metric="auc",
            callbacks=[early_stopping(100), log_evaluation(200)],
        )
        oof_adv[va_idx] = model.predict_proba(X_adv[va_idx])[:, 1]

    adv_auc = float(roc_auc_score(y_adv, oof_adv))
    p_testlike = oof_adv[: len(X_train)]
    weights = p_testlike / (float(np.mean(p_testlike)) + 1e-12)
    weights = np.clip(weights, args.adv_weight_min, args.adv_weight_max)
    weights = weights / (float(np.mean(weights)) + 1e-12)

    summary = {
        "adversarial_auc_train_vs_test": adv_auc,
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "testlike_prob_mean_train": float(np.mean(p_testlike)),
    }
    return weights.astype(np.float32), summary


def build_error_features(
    X: pd.DataFrame,
    anchor_pred: np.ndarray,
    diversity_pred: np.ndarray,
    dynamic_pred: np.ndarray,
    dynamic_weight: np.ndarray,
    uncertainty: np.ndarray,
    disagreement_norm: np.ndarray,
    adv_weights: np.ndarray,
) -> np.ndarray:
    cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "charge_per_tenure",
        "charges_gap_abs",
        "is_month_to_month",
        "is_electronic_check",
        "is_new_customer",
        "is_long_customer",
    ]
    base = X[cols].to_numpy(dtype=np.float32)
    meta = np.column_stack(
        [
            anchor_pred.astype(np.float32),
            diversity_pred.astype(np.float32),
            dynamic_pred.astype(np.float32),
            dynamic_weight.astype(np.float32),
            uncertainty.astype(np.float32),
            disagreement_norm.astype(np.float32),
            np.abs(anchor_pred - diversity_pred).astype(np.float32),
            (anchor_pred * diversity_pred).astype(np.float32),
            adv_weights.astype(np.float32),
        ]
    )
    return np.hstack([base, meta]).astype(np.float32)


def fit_error_correction(
    err_X_train: np.ndarray,
    y: np.ndarray,
    dynamic_oof: np.ndarray,
    err_X_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    residual = y - dynamic_oof
    abs_residual = np.abs(residual)
    threshold = float(np.quantile(abs_residual, args.error_quantile))
    hard_mask = abs_residual >= threshold
    hard_fraction = float(np.mean(hard_mask))

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state + 73)
    corr_oof = np.zeros(len(y), dtype=np.float64)
    corr_test = np.zeros(len(err_X_test), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(err_X_train, y), start=1):
        hard_tr_idx = tr_idx[hard_mask[tr_idx]]
        if len(hard_tr_idx) < max(256, int(0.05 * len(tr_idx))):
            hard_tr_idx = tr_idx

        model = LGBMRegressor(
            n_estimators=args.error_estimators,
            learning_rate=args.error_lr,
            num_leaves=48,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.2,
            n_jobs=args.n_jobs,
            random_state=args.random_state * 100 + fold,
            verbosity=-1,
        )
        model.fit(
            err_X_train[hard_tr_idx],
            residual[hard_tr_idx],
            eval_set=[(err_X_train[va_idx], residual[va_idx])],
            eval_metric="l2",
            callbacks=[early_stopping(80), log_evaluation(200)],
        )
        corr_oof[va_idx] = model.predict(err_X_train[va_idx])
        corr_test += model.predict(err_X_test) / args.n_splits

    summary = {
        "hard_residual_threshold": threshold,
        "hard_sample_fraction": hard_fraction,
        "correction_oof_std": float(np.std(corr_oof)),
        "correction_test_std": float(np.std(corr_test)),
    }
    return corr_oof, corr_test, summary


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
    cat_task_type, xgb_device, device_note = resolve_devices(args.device_mode, args.gpu_devices)
    print(f"Resolved devices: catboost={cat_task_type}, xgboost={xgb_device} ({device_note})")

    train_df = add_features(pd.read_csv(args.train))
    test_df = add_features(pd.read_csv(args.test))

    all_features, cat_features, num_features = build_feature_lists(train_df)
    X_all = train_df[all_features]
    y_all = map_target(train_df[TARGET_COLUMN])
    X_test = test_df[all_features]

    X, y = select_training_subset(X_all, y_all, args.max_train_rows, args.random_state)
    if len(X) != len(X_all):
        print(f"Using subset: {len(X)} / {len(X_all)} rows")

    adv_weights, adv_summary = compute_adversarial_weights(X, X_test, cat_features, num_features, args)
    print(
        "Adversarial weights summary | "
        f"auc={adv_summary['adversarial_auc_train_vs_test']:.6f} "
        f"mean={adv_summary['weight_mean']:.4f} std={adv_summary['weight_std']:.4f} "
        f"min={adv_summary['weight_min']:.4f} max={adv_summary['weight_max']:.4f}"
    )

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    oof_cat = np.zeros(len(X), dtype=np.float64)
    oof_lgb = np.zeros(len(X), dtype=np.float64)
    oof_xgb = np.zeros(len(X), dtype=np.float64)
    test_cat = np.zeros(len(X_test), dtype=np.float64)
    test_lgb = np.zeros(len(X_test), dtype=np.float64)
    test_xgb = np.zeros(len(X_test), dtype=np.float64)
    fold_metrics: list[dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n===== Fold {fold}/{args.n_splits} =====")
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]
        w_tr = adv_weights[tr_idx]
        seed = args.random_state * 100 + fold

        va_cat, te_cat = train_catboost_fold(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_test,
            cat_features,
            w_tr,
            seed,
            cat_task_type,
            args.gpu_devices,
            args,
        )
        va_lgb, te_lgb = train_lgb_fold(X_tr, y_tr, X_va, y_va, X_test, cat_features, w_tr, seed, args)

        tr_cat_te, va_cat_te, te_cat_te = fit_target_encoder(
            X_tr,
            y_tr,
            X_va,
            X_test,
            cat_features,
            args.te_cv,
            args.te_smooth,
            seed,
        )
        tr_num = X_tr[num_features].to_numpy(dtype=np.float32)
        va_num = X_va[num_features].to_numpy(dtype=np.float32)
        te_num = X_test[num_features].to_numpy(dtype=np.float32)
        Xtr_te = np.hstack([tr_num, tr_cat_te])
        Xva_te = np.hstack([va_num, va_cat_te])
        Xte_te = np.hstack([te_num, te_cat_te])

        va_xgb, te_xgb = train_xgb_fold(
            Xtr_te,
            y_tr.to_numpy(),
            Xva_te,
            y_va.to_numpy(),
            Xte_te,
            w_tr,
            seed,
            xgb_device,
            args,
        )

        oof_cat[va_idx] = va_cat
        oof_lgb[va_idx] = va_lgb
        oof_xgb[va_idx] = va_xgb
        test_cat += te_cat / args.n_splits
        test_lgb += te_lgb / args.n_splits
        test_xgb += te_xgb / args.n_splits

        fold_metric = {
            "fold": float(fold),
            "cat_auc": float(roc_auc_score(y_va, va_cat)),
            "lgb_auc": float(roc_auc_score(y_va, va_lgb)),
            "xgb_auc": float(roc_auc_score(y_va, va_xgb)),
        }
        fold_metrics.append(fold_metric)
        print(
            f"Fold {fold} AUCs | Cat={fold_metric['cat_auc']:.6f} "
            f"| LGB={fold_metric['lgb_auc']:.6f} | XGB={fold_metric['xgb_auc']:.6f}"
        )

    y_np = y.to_numpy()
    cat_auc = float(roc_auc_score(y_np, oof_cat))
    lgb_auc = float(roc_auc_score(y_np, oof_lgb))
    xgb_auc = float(roc_auc_score(y_np, oof_xgb))

    anchor_weight_cat, anchor_auc = optimize_anchor_weight(y_np, oof_cat, oof_lgb)
    anchor_oof = anchor_weight_cat * oof_cat + (1.0 - anchor_weight_cat) * oof_lgb
    anchor_test = anchor_weight_cat * test_cat + (1.0 - anchor_weight_cat) * test_lgb

    dynamic_oof, dynamic_weight_oof, uncertainty_oof, disagreement_oof = build_dynamic_blend(
        anchor_oof,
        oof_xgb,
        args.dynamic_base_weight,
        args.dynamic_uncertainty_weight,
        args.dynamic_disagreement_weight,
        args.dynamic_max_weight,
    )
    dynamic_test, dynamic_weight_test, uncertainty_test, disagreement_test = build_dynamic_blend(
        anchor_test,
        test_xgb,
        args.dynamic_base_weight,
        args.dynamic_uncertainty_weight,
        args.dynamic_disagreement_weight,
        args.dynamic_max_weight,
    )
    dynamic_auc = float(roc_auc_score(y_np, dynamic_oof))

    err_X_train = build_error_features(
        X,
        anchor_oof,
        oof_xgb,
        dynamic_oof,
        dynamic_weight_oof,
        uncertainty_oof,
        disagreement_oof,
        adv_weights.astype(np.float64),
    )
    err_X_test = build_error_features(
        X_test,
        anchor_test,
        test_xgb,
        dynamic_test,
        dynamic_weight_test,
        uncertainty_test,
        disagreement_test,
        np.ones(len(X_test), dtype=np.float64),
    )
    corr_oof, corr_test, corr_summary = fit_error_correction(err_X_train, y_np, dynamic_oof, err_X_test, args)

    final_oof = np.clip(dynamic_oof + args.error_corr_scale * corr_oof, 0.0, 1.0)
    final_test = np.clip(dynamic_test + args.error_corr_scale * corr_test, 0.0, 1.0)
    final_auc = float(roc_auc_score(y_np, final_oof))

    submission_df = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: final_test})
    validate_submission(submission_df, len(test_df))

    submission_path = Path(args.submission)
    metrics_path = Path(args.metrics)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

    metrics = {
        "cv_splits": args.n_splits,
        "random_state": args.random_state,
        "device_mode_requested": args.device_mode,
        "device_resolved": {
            "catboost_task_type": cat_task_type,
            "xgboost_device": xgb_device,
            "note": device_note,
        },
        "max_train_rows": args.max_train_rows,
        "used_train_rows": int(len(X)),
        "fold_metrics": fold_metrics,
        "adversarial_weighting": adv_summary,
        "per_model_oof_auc": {
            "cat": cat_auc,
            "lgb": lgb_auc,
            "xgb": xgb_auc,
        },
        "anchor_blend": {
            "cat_weight": anchor_weight_cat,
            "lgb_weight": 1.0 - anchor_weight_cat,
            "oof_auc": anchor_auc,
        },
        "dynamic_blend": {
            "base_weight": args.dynamic_base_weight,
            "uncertainty_weight": args.dynamic_uncertainty_weight,
            "disagreement_weight": args.dynamic_disagreement_weight,
            "max_weight": args.dynamic_max_weight,
            "oof_auc": dynamic_auc,
            "dynamic_weight_oof_mean": float(np.mean(dynamic_weight_oof)),
            "dynamic_weight_oof_std": float(np.std(dynamic_weight_oof)),
        },
        "error_correction": {
            **corr_summary,
            "correction_scale": args.error_corr_scale,
        },
        "final_oof_auc": final_auc,
        "train_rows_total": int(len(X_all)),
        "test_rows": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n===== V7 Summary =====")
    print(f"Cat OOF AUC: {cat_auc:.6f}")
    print(f"LGB OOF AUC: {lgb_auc:.6f}")
    print(f"XGB OOF AUC: {xgb_auc:.6f}")
    print(f"Anchor OOF AUC: {anchor_auc:.6f} (cat={anchor_weight_cat:.2f}, lgb={1.0-anchor_weight_cat:.2f})")
    print(f"Dynamic OOF AUC: {dynamic_auc:.6f}")
    print(f"Final OOF AUC (with error correction): {final_auc:.6f}")
    print(f"Submission written to: {submission_path}")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
