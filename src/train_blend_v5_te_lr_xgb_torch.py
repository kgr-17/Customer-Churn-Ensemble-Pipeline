import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
        description="Train hybrid ensemble: TargetEncoder + LogisticRegression + XGBoost + PyTorch MLP"
    )
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--submission",
        default="outputs/submission_blend_v5_te_lr_xgb_torch.csv",
        help="Output submission CSV",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/metrics_blend_v5_te_lr_xgb_torch.json",
        help="Output metrics JSON",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="CV random seed")
    parser.add_argument(
        "--device-mode",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Device mode for XGBoost/Torch (auto=try GPU then CPU)",
    )
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional stratified row cap for faster iteration")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Thread count for CPU parts")
    parser.add_argument("--te-cv", type=int, default=5, help="TargetEncoder internal cross-fitting folds")
    parser.add_argument("--te-smooth", default="auto", help="TargetEncoder smooth parameter (float or 'auto')")

    parser.add_argument("--xgb-estimators", type=int, default=2500, help="XGBoost n_estimators")
    parser.add_argument("--xgb-lr", type=float, default=0.03, help="XGBoost learning rate")
    parser.add_argument("--xgb-depth", type=int, default=6, help="XGBoost max depth")
    parser.add_argument("--xgb-min-child-weight", type=float, default=4.0, help="XGBoost min_child_weight")
    parser.add_argument("--xgb-es-rounds", type=int, default=250, help="XGBoost early stopping rounds")

    parser.add_argument("--mlp-epochs", type=int, default=10, help="Max MLP epochs")
    parser.add_argument("--mlp-batch-size", type=int, default=8192, help="MLP batch size")
    parser.add_argument("--mlp-lr", type=float, default=1e-3, help="MLP learning rate")
    parser.add_argument("--mlp-hidden1", type=int, default=256, help="MLP first hidden layer size")
    parser.add_argument("--mlp-hidden2", type=int, default=128, help="MLP second hidden layer size")
    parser.add_argument("--mlp-dropout", type=float, default=0.10, help="MLP dropout")
    parser.add_argument("--mlp-patience", type=int, default=3, help="Early-stop patience for MLP")
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


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
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
    return cat_features, num_features


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


def detect_xgb_gpu() -> tuple[bool, str]:
    X_small = np.array(
        [
            [0.0, 0.1],
            [1.0, 0.3],
            [0.0, 0.2],
            [1.0, 0.4],
            [0.0, 0.5],
            [1.0, 0.7],
        ],
        dtype=np.float32,
    )
    y_small = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    try:
        model = XGBClassifier(
            n_estimators=10,
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


def resolve_devices(device_mode: str) -> tuple[str, torch.device, str]:
    torch_gpu_ok = torch.cuda.is_available()
    if device_mode == "cpu":
        return "cpu", torch.device("cpu"), "forced_cpu"

    xgb_gpu_ok, xgb_reason = detect_xgb_gpu()
    if device_mode == "gpu":
        if not xgb_gpu_ok:
            raise RuntimeError(f"Requested GPU mode but XGBoost GPU is unavailable: {xgb_reason}")
        if not torch_gpu_ok:
            raise RuntimeError("Requested GPU mode but torch.cuda is not available")
        return "cuda", torch.device("cuda"), "forced_gpu"

    # auto mode
    if xgb_gpu_ok and torch_gpu_ok:
        return "cuda", torch.device("cuda"), "auto_gpu"
    return "cpu", torch.device("cpu"), f"auto_cpu_fallback: xgb={xgb_reason}, torch={torch_gpu_ok}"


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
    encoder = TargetEncoder(
        categories="auto",
        target_type="binary",
        smooth=smooth_value,
        cv=te_cv,
        random_state=seed,
    )
    tr_enc = encoder.fit_transform(X_tr[cat_features], y_tr)
    va_enc = encoder.transform(X_va[cat_features])
    test_enc = encoder.transform(X_test[cat_features])
    return np.asarray(tr_enc, dtype=np.float32), np.asarray(va_enc, dtype=np.float32), np.asarray(test_enc, dtype=np.float32)


def build_matrices(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list[str],
    num_features: list[str],
    te_cv: int,
    te_smooth: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr_cat, va_cat, test_cat = fit_target_encoder(
        X_tr, y_tr, X_va, X_test, cat_features, te_cv=te_cv, te_smooth=te_smooth, seed=seed
    )
    tr_num = X_tr[num_features].to_numpy(dtype=np.float32)
    va_num = X_va[num_features].to_numpy(dtype=np.float32)
    test_num = X_test[num_features].to_numpy(dtype=np.float32)
    Xtr = np.hstack([tr_num, tr_cat])
    Xva = np.hstack([va_num, va_cat])
    Xte = np.hstack([test_num, test_cat])
    return Xtr, Xva, Xte


def train_logistic(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    Xte: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xva_sc = scaler.transform(Xva)
    Xte_sc = scaler.transform(Xte)
    model = LogisticRegression(
        C=0.8,
        solver="lbfgs",
        max_iter=600,
        random_state=random_state,
    )
    model.fit(Xtr_sc, ytr)
    va_pred = model.predict_proba(Xva_sc)[:, 1]
    te_pred = model.predict_proba(Xte_sc)[:, 1]
    return va_pred, te_pred


def train_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    seed: int,
    device: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=args.xgb_estimators,
        learning_rate=args.xgb_lr,
        max_depth=args.xgb_depth,
        min_child_weight=args.xgb_min_child_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=args.n_jobs,
        early_stopping_rounds=args.xgb_es_rounds,
    )
    try:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    except XGBoostError as exc:
        raise RuntimeError(f"XGBoost failed on device={device}: {exc}") from exc
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_torch_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr).astype(np.float32)
    Xva_sc = scaler.transform(Xva).astype(np.float32)
    Xte_sc = scaler.transform(Xte).astype(np.float32)

    tr_ds = TensorDataset(torch.from_numpy(Xtr_sc), torch.from_numpy(ytr.astype(np.float32)))
    tr_loader = DataLoader(tr_ds, batch_size=args.mlp_batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model = TorchMLP(
        input_dim=Xtr_sc.shape[1],
        hidden1=args.mlp_hidden1,
        hidden2=args.mlp_hidden2,
        dropout=args.mlp_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xva_t = torch.from_numpy(Xva_sc).to(device)
    Xte_t = torch.from_numpy(Xte_sc).to(device)

    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0

    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
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
        va_auc = float(roc_auc_score(yva, va_pred))
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
    va_pred = 1.0 / (1.0 + np.exp(-va_logits))
    te_pred = 1.0 / (1.0 + np.exp(-te_logits))
    return va_pred, te_pred


def optimize_three_model_weights(
    y: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> tuple[dict[str, float], float]:
    best_auc = -1.0
    best_weights = {"lr": 1 / 3, "xgb": 1 / 3, "mlp": 1 / 3}
    # Coarse grid first for speed/robustness.
    for w1 in np.linspace(0.0, 1.0, 21):
        for w2 in np.linspace(0.0, 1.0 - w1, int((1.0 - w1) / 0.05) + 1):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            blend = w1 * p1 + w2 * p2 + w3 * p3
            auc = float(roc_auc_score(y, blend))
            if auc > best_auc:
                best_auc = auc
                best_weights = {"lr": float(w1), "xgb": float(w2), "mlp": float(w3)}
    return best_weights, best_auc


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
    xgb_device, torch_device, device_note = resolve_devices(args.device_mode)
    print(f"Resolved devices: xgboost={xgb_device}, torch={torch_device.type} ({device_note})")

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    cat_features, num_features = build_feature_lists(train_df)
    all_features = cat_features + num_features

    y_all = map_target(train_df[TARGET_COLUMN])
    X_all = train_df[all_features]
    X_test = test_df[all_features]

    X, y = select_training_subset(X_all, y_all, args.max_train_rows, args.random_state)
    if len(X) != len(X_all):
        print(f"Using subset: {len(X)} / {len(X_all)} rows")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    oof_lr = np.zeros(len(X), dtype=np.float64)
    oof_xgb = np.zeros(len(X), dtype=np.float64)
    oof_mlp = np.zeros(len(X), dtype=np.float64)
    test_lr = np.zeros(len(X_test), dtype=np.float64)
    test_xgb = np.zeros(len(X_test), dtype=np.float64)
    test_mlp = np.zeros(len(X_test), dtype=np.float64)
    fold_metrics: list[dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n===== Fold {fold}/{args.n_splits} =====")
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        Xtr_mat, Xva_mat, Xte_mat = build_matrices(
            X_tr, y_tr, X_va, X_test, cat_features, num_features, te_cv=args.te_cv, te_smooth=args.te_smooth, seed=args.random_state + fold
        )

        va_lr, te_lr = train_logistic(
            Xtr_mat,
            y_tr.to_numpy(),
            Xva_mat,
            Xte_mat,
            random_state=args.random_state + fold,
        )
        va_xgb, te_xgb = train_xgb(
            Xtr_mat,
            y_tr.to_numpy(),
            Xva_mat,
            y_va.to_numpy(),
            Xte_mat,
            seed=args.random_state + fold,
            device=xgb_device,
            args=args,
        )
        va_mlp, te_mlp = train_torch_mlp(
            Xtr_mat,
            y_tr.to_numpy(),
            Xva_mat,
            y_va.to_numpy(),
            Xte_mat,
            device=torch_device,
            seed=args.random_state + fold,
            args=args,
        )

        oof_lr[va_idx] = va_lr
        oof_xgb[va_idx] = va_xgb
        oof_mlp[va_idx] = va_mlp
        test_lr += te_lr / args.n_splits
        test_xgb += te_xgb / args.n_splits
        test_mlp += te_mlp / args.n_splits

        fold_auc_lr = float(roc_auc_score(y_va, va_lr))
        fold_auc_xgb = float(roc_auc_score(y_va, va_xgb))
        fold_auc_mlp = float(roc_auc_score(y_va, va_mlp))
        fold_metrics.append(
            {
                "fold": float(fold),
                "lr_auc": fold_auc_lr,
                "xgb_auc": fold_auc_xgb,
                "mlp_auc": fold_auc_mlp,
            }
        )
        print(
            f"Fold {fold} AUCs | LR={fold_auc_lr:.6f} | XGB={fold_auc_xgb:.6f} | MLP={fold_auc_mlp:.6f}"
        )

    y_np = y.to_numpy()
    lr_oof_auc = float(roc_auc_score(y_np, oof_lr))
    xgb_oof_auc = float(roc_auc_score(y_np, oof_xgb))
    mlp_oof_auc = float(roc_auc_score(y_np, oof_mlp))
    best_weights, best_blend_auc = optimize_three_model_weights(y_np, oof_lr, oof_xgb, oof_mlp)

    final_test_pred = (
        best_weights["lr"] * test_lr + best_weights["xgb"] * test_xgb + best_weights["mlp"] * test_mlp
    )

    submission_df = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: final_test_pred})
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
        "xgb_device_resolved": xgb_device,
        "torch_device_resolved": torch_device.type,
        "device_note": device_note,
        "max_train_rows": args.max_train_rows,
        "used_train_rows": int(len(X)),
        "fold_metrics": fold_metrics,
        "lr_oof_auc": lr_oof_auc,
        "xgb_oof_auc": xgb_oof_auc,
        "mlp_oof_auc": mlp_oof_auc,
        "best_blend_weights": best_weights,
        "best_blend_oof_auc": best_blend_auc,
        "train_rows_total": int(len(X_all)),
        "test_rows": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n===== Hybrid Summary =====")
    print(f"LR OOF AUC: {lr_oof_auc:.6f}")
    print(f"XGB OOF AUC: {xgb_oof_auc:.6f}")
    print(f"MLP OOF AUC: {mlp_oof_auc:.6f}")
    print(
        f"Blend OOF AUC: {best_blend_auc:.6f} "
        f"(w_lr={best_weights['lr']:.2f}, w_xgb={best_weights['xgb']:.2f}, w_mlp={best_weights['mlp']:.2f})"
    )
    print(f"Submission written to: {submission_path}")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
