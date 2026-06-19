import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostError
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train super GPU blend: CatBoost + LightGBM + XGBoost + Logistic + Torch MLP"
    )
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--submission",
        default="outputs/submission_blend_v6_super_gpu.csv",
        help="Output submission CSV",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/metrics_blend_v6_super_gpu.json",
        help="Output metrics JSON",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="CV random seed")
    parser.add_argument(
        "--device-mode",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Device mode for CatBoost/XGBoost/Torch (auto=try GPU then CPU)",
    )
    parser.add_argument("--gpu-devices", default="0", help="CatBoost devices when using GPU mode")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional stratified row cap")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Thread count for CPU models")

    parser.add_argument("--te-cv", type=int, default=5, help="TargetEncoder cross-fitting folds")
    parser.add_argument("--te-smooth", default="auto", help="TargetEncoder smooth (float or 'auto')")

    parser.add_argument("--cat-iterations", type=int, default=2800, help="CatBoost iterations")
    parser.add_argument("--cat-lr", type=float, default=0.03, help="CatBoost learning rate")
    parser.add_argument("--cat-depth", type=int, default=8, help="CatBoost depth")
    parser.add_argument("--cat-es-rounds", type=int, default=250, help="CatBoost early stopping rounds")

    parser.add_argument("--lgb-estimators", type=int, default=4000, help="LightGBM n_estimators")
    parser.add_argument("--lgb-lr", type=float, default=0.02, help="LightGBM learning rate")
    parser.add_argument("--lgb-es-rounds", type=int, default=300, help="LightGBM early stopping rounds")

    parser.add_argument("--xgb-estimators", type=int, default=2400, help="XGBoost n_estimators")
    parser.add_argument("--xgb-lr", type=float, default=0.03, help="XGBoost learning rate")
    parser.add_argument("--xgb-depth", type=int, default=6, help="XGBoost max depth")
    parser.add_argument("--xgb-es-rounds", type=int, default=250, help="XGBoost early stopping rounds")

    parser.add_argument("--mlp-epochs", type=int, default=6, help="Max MLP epochs")
    parser.add_argument("--mlp-batch-size", type=int, default=8192, help="MLP batch size")
    parser.add_argument("--mlp-lr", type=float, default=1e-3, help="MLP learning rate")
    parser.add_argument("--mlp-hidden1", type=int, default=256, help="MLP hidden layer 1 size")
    parser.add_argument("--mlp-hidden2", type=int, default=128, help="MLP hidden layer 2 size")
    parser.add_argument("--mlp-dropout", type=float, default=0.10, help="MLP dropout")
    parser.add_argument("--mlp-patience", type=int, default=2, help="MLP early-stop patience")

    parser.add_argument("--blend-samples", type=int, default=7000, help="Random samples for blend search")
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


def resolve_devices(device_mode: str, gpu_devices: str) -> tuple[str, str, torch.device, str]:
    if device_mode == "cpu":
        return "CPU", "cpu", torch.device("cpu"), "forced_cpu"

    cat_ok, cat_reason = detect_catboost_gpu(gpu_devices)
    xgb_ok, xgb_reason = detect_xgb_gpu()
    torch_ok = torch.cuda.is_available()

    if device_mode == "gpu":
        if not cat_ok:
            raise RuntimeError(f"Requested GPU mode but CatBoost GPU unavailable: {cat_reason}")
        if not xgb_ok:
            raise RuntimeError(f"Requested GPU mode but XGBoost GPU unavailable: {xgb_reason}")
        if not torch_ok:
            raise RuntimeError("Requested GPU mode but torch.cuda is unavailable")
        return "GPU", "cuda", torch.device("cuda"), "forced_gpu"

    # auto mode
    if cat_ok and xgb_ok and torch_ok:
        return "GPU", "cuda", torch.device("cuda"), "auto_gpu"
    return (
        "CPU",
        "cpu",
        torch.device("cpu"),
        f"auto_cpu_fallback: cat={cat_reason}, xgb={xgb_reason}, torch={torch_ok}",
    )


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr).astype(np.float32)
    Xva = scaler.transform(X_va).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_tr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=args.mlp_batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model = TorchMLP(Xtr.shape[1], args.mlp_hidden1, args.mlp_hidden2, args.mlp_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xva_t = torch.from_numpy(Xva).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)

    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0

    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(Xva_t).detach().cpu().numpy()
        va_pred = 1.0 / (1.0 + np.exp(-va_logits))
        va_auc = float(roc_auc_score(y_va, va_pred))
        print(f"MLP epoch {epoch}/{args.mlp_epochs} | val_auc={va_auc:.6f}")

        if va_auc > best_auc + 1e-6:
            best_auc = va_auc
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.mlp_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    with torch.no_grad():
        va_logits = model(Xva_t).detach().cpu().numpy()
        te_logits = model(Xte_t).detach().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-va_logits)), 1.0 / (1.0 + np.exp(-te_logits))


def train_catboost_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
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
        model.fit(X_tr, y_tr, cat_features=cat_indices, eval_set=(X_va, y_va), use_best_model=True)
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
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except XGBoostError as exc:
        raise RuntimeError(f"XGBoost failed on {xgb_device}: {exc}") from exc
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_test)[:, 1]


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def optimize_prob_blend(y: np.ndarray, preds: np.ndarray, n_samples: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n_models = preds.shape[1]
    best_auc = -1.0
    best_w = np.full(n_models, 1.0 / n_models, dtype=np.float64)

    # deterministic starters
    starter_weights = [best_w]
    for i in range(n_models):
        w = np.zeros(n_models, dtype=np.float64)
        w[i] = 1.0
        starter_weights.append(w)
    for _ in range(n_samples):
        starter_weights.append(rng.dirichlet(np.ones(n_models)))

    for w in starter_weights:
        blend = preds @ w
        auc = float(roc_auc_score(y, blend))
        if auc > best_auc:
            best_auc = auc
            best_w = w
    return best_w, best_auc


def optimize_logit_blend(y: np.ndarray, preds: np.ndarray, n_samples: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed + 7)
    logits = safe_logit(preds)
    n_models = logits.shape[1]
    best_auc = -1.0
    best_w = np.ones(n_models, dtype=np.float64)

    candidates: list[np.ndarray] = []
    candidates.append(np.ones(n_models, dtype=np.float64))
    for i in range(n_models):
        w = np.zeros(n_models, dtype=np.float64)
        w[i] = 1.0
        candidates.append(w)
    for _ in range(n_samples):
        candidates.append(rng.uniform(-0.6, 1.4, size=n_models))

    for w in candidates:
        pred = sigmoid(logits @ w)
        auc = float(roc_auc_score(y, pred))
        if auc > best_auc:
            best_auc = auc
            best_w = w
    return best_w, best_auc


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
    all_features, cat_features, num_features = build_feature_lists(add_features(pd.read_csv(args.train).head(2)))
    del all_features  # only schema check for now

    cat_task_type, xgb_device, torch_device, device_note = resolve_devices(args.device_mode, args.gpu_devices)
    print(
        f"Resolved devices: catboost={cat_task_type}, xgboost={xgb_device}, "
        f"torch={torch_device.type} ({device_note})"
    )

    train_df = add_features(pd.read_csv(args.train))
    test_df = add_features(pd.read_csv(args.test))

    all_features, cat_features, num_features = build_feature_lists(train_df)
    X_all = train_df[all_features]
    y_all = map_target(train_df[TARGET_COLUMN])
    X_test = test_df[all_features]

    X, y = select_training_subset(X_all, y_all, args.max_train_rows, args.random_state)
    if len(X) != len(X_all):
        print(f"Using subset: {len(X)} / {len(X_all)} rows")

    n_models = 5
    model_names = ["cat", "lgb", "xgb", "lr", "mlp"]
    oof_mat = np.zeros((len(X), n_models), dtype=np.float64)
    test_mat = np.zeros((len(X_test), n_models), dtype=np.float64)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    fold_metrics: list[dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n===== Fold {fold}/{args.n_splits} =====")
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        # CatBoost + LightGBM directly on mixed feature table.
        seed = args.random_state * 100 + fold
        va_cat, te_cat = train_catboost_fold(
            X_tr, y_tr, X_va, y_va, X_test, cat_features, seed, cat_task_type, args.gpu_devices, args
        )
        va_lgb, te_lgb = train_lgb_fold(X_tr, y_tr, X_va, y_va, X_test, cat_features, seed, args)

        # TE matrix for LR/XGB/MLP.
        tr_cat_te, va_cat_te, te_cat_te = fit_target_encoder(
            X_tr, y_tr, X_va, X_test, cat_features, args.te_cv, args.te_smooth, seed
        )
        tr_num = X_tr[num_features].to_numpy(dtype=np.float32)
        va_num = X_va[num_features].to_numpy(dtype=np.float32)
        te_num = X_test[num_features].to_numpy(dtype=np.float32)
        Xtr_te = np.hstack([tr_num, tr_cat_te])
        Xva_te = np.hstack([va_num, va_cat_te])
        Xte_te = np.hstack([te_num, te_cat_te])

        scaler = StandardScaler()
        Xtr_lr = scaler.fit_transform(Xtr_te)
        Xva_lr = scaler.transform(Xva_te)
        Xte_lr = scaler.transform(Xte_te)
        lr = LogisticRegression(C=0.8, solver="lbfgs", max_iter=600, random_state=seed)
        lr.fit(Xtr_lr, y_tr.to_numpy())
        va_lr = lr.predict_proba(Xva_lr)[:, 1]
        te_lr = lr.predict_proba(Xte_lr)[:, 1]

        va_xgb, te_xgb = train_xgb_fold(
            Xtr_te, y_tr.to_numpy(), Xva_te, y_va.to_numpy(), Xte_te, seed, xgb_device, args
        )
        va_mlp, te_mlp = train_mlp(
            Xtr_te, y_tr.to_numpy(), Xva_te, y_va.to_numpy(), Xte_te, torch_device, seed, args
        )

        fold_preds = np.column_stack([va_cat, va_lgb, va_xgb, va_lr, va_mlp])
        oof_mat[va_idx, :] = fold_preds
        test_mat += np.column_stack([te_cat, te_lgb, te_xgb, te_lr, te_mlp]) / args.n_splits

        fold_metric = {
            "fold": float(fold),
            "cat_auc": float(roc_auc_score(y_va, va_cat)),
            "lgb_auc": float(roc_auc_score(y_va, va_lgb)),
            "xgb_auc": float(roc_auc_score(y_va, va_xgb)),
            "lr_auc": float(roc_auc_score(y_va, va_lr)),
            "mlp_auc": float(roc_auc_score(y_va, va_mlp)),
        }
        fold_metrics.append(fold_metric)
        print(
            f"Fold {fold} AUCs | Cat={fold_metric['cat_auc']:.6f} | LGB={fold_metric['lgb_auc']:.6f} | "
            f"XGB={fold_metric['xgb_auc']:.6f} | LR={fold_metric['lr_auc']:.6f} | MLP={fold_metric['mlp_auc']:.6f}"
        )

    y_np = y.to_numpy()
    per_model_auc = {name: float(roc_auc_score(y_np, oof_mat[:, i])) for i, name in enumerate(model_names)}

    prob_w, prob_auc = optimize_prob_blend(y_np, oof_mat, args.blend_samples, args.random_state)
    logit_w, logit_auc = optimize_logit_blend(y_np, oof_mat, args.blend_samples, args.random_state)

    if logit_auc > prob_auc:
        chosen_strategy = "logit_weighted"
        test_pred = sigmoid(safe_logit(test_mat) @ logit_w)
        chosen_auc = logit_auc
    else:
        chosen_strategy = "probability_weighted"
        test_pred = test_mat @ prob_w
        chosen_auc = prob_auc

    submission_df = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: test_pred})
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
            "torch_device": torch_device.type,
            "note": device_note,
        },
        "max_train_rows": args.max_train_rows,
        "used_train_rows": int(len(X)),
        "fold_metrics": fold_metrics,
        "per_model_oof_auc": per_model_auc,
        "probability_blend": {
            "weights": {name: float(prob_w[i]) for i, name in enumerate(model_names)},
            "oof_auc": float(prob_auc),
        },
        "logit_blend": {
            "weights": {name: float(logit_w[i]) for i, name in enumerate(model_names)},
            "oof_auc": float(logit_auc),
        },
        "chosen_strategy": chosen_strategy,
        "chosen_oof_auc": float(chosen_auc),
        "train_rows_total": int(len(X_all)),
        "test_rows": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n===== Super Blend Summary =====")
    for name in model_names:
        print(f"{name.upper()} OOF AUC: {per_model_auc[name]:.6f}")
    print(f"Probability blend OOF AUC: {prob_auc:.6f}")
    print(f"Logit blend OOF AUC: {logit_auc:.6f}")
    print(f"Chosen strategy: {chosen_strategy} | OOF AUC: {chosen_auc:.6f}")
    print(f"Submission written to: {submission_path}")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
