import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_list(text: str) -> list[str]:
    parts = [part.strip() for part in text.split(",")]
    values = [part for part in parts if part]
    if not values:
        raise ValueError("At least one value is required")
    return values


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in parse_list(text)]


def resolve_compute_backend(device_mode: str) -> tuple[str | None, str]:
    if device_mode == "cpu":
        return None, "cpu_numpy(forced)"

    try:
        import torch  # type: ignore
    except Exception as exc:
        if device_mode == "gpu":
            raise RuntimeError(f"Requested GPU mode but torch import failed: {exc}") from exc
        return None, f"cpu_numpy(auto_torch_import_failed: {exc})"

    if device_mode == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU mode but torch.cuda is unavailable")
        return "cuda", "gpu_cuda(forced)"

    if torch.cuda.is_available():
        return "cuda", "gpu_cuda(auto)"
    return None, "cpu_numpy(auto_no_cuda)"


def resolve_target_column(df: pd.DataFrame, target_column: str, id_column: str) -> str:
    if target_column in df.columns:
        return target_column

    aliases = ["Churn", "target", "pred", "prediction"]
    available = [name for name in aliases if name in df.columns and name != id_column]
    if len(available) == 1:
        return available[0]

    raise ValueError(
        f"Could not resolve prediction column. Looked for '{target_column}' and aliases {aliases}. "
        f"Columns were: {list(df.columns)}"
    )


def load_submissions(
    input_paths: list[Path],
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    merged: pd.DataFrame | None = None
    pred_columns: list[str] = []

    for idx, path in enumerate(input_paths):
        df = pd.read_csv(path)
        if id_column not in df.columns:
            raise ValueError(f"Missing id column '{id_column}' in {path}")
        pred_col_src = resolve_target_column(df, target_column, id_column)
        pred_col = f"pred_{idx}"
        frame = df[[id_column, pred_col_src]].rename(columns={pred_col_src: pred_col})
        pred_columns.append(pred_col)

        if merged is None:
            merged = frame
            continue

        before_rows = len(merged)
        merged = merged.merge(frame, on=id_column, how="inner")
        if len(merged) != before_rows:
            raise ValueError(
                f"Row mismatch while merging {path}. Expected {before_rows} rows, got {len(merged)}"
            )

    if merged is None:
        raise ValueError("No submissions were loaded")
    return merged, pred_columns


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if np.isclose(total, 0.0):
        raise ValueError("Sum of weights must not be zero")
    return weights / total


def weighted_average(pred_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = normalize_weights(weights)
    return pred_matrix @ w


def weighted_average_torch(pred_matrix: np.ndarray, weights: np.ndarray, torch_device: str) -> np.ndarray:
    import torch  # type: ignore

    pred_t = torch.as_tensor(pred_matrix, dtype=torch.float32, device=torch_device)
    w_t = torch.as_tensor(normalize_weights(weights), dtype=torch.float32, device=torch_device)
    return (pred_t @ w_t).detach().cpu().numpy().astype(np.float64)


def weighted_rank_average(pred_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_rows = pred_matrix.shape[0]
    ranks = np.empty_like(pred_matrix, dtype=np.float64)

    # Convert each model's predictions to percent ranks for robust rank blending.
    for col in range(pred_matrix.shape[1]):
        order = np.argsort(pred_matrix[:, col], kind="mergesort")
        rank_values = np.empty(n_rows, dtype=np.float64)
        rank_values[order] = np.arange(1, n_rows + 1, dtype=np.float64)
        ranks[:, col] = rank_values / (n_rows + 1.0)

    w = normalize_weights(weights)
    return ranks @ w


def weighted_rank_average_torch(pred_matrix: np.ndarray, weights: np.ndarray, torch_device: str) -> np.ndarray:
    import torch  # type: ignore

    pred_t = torch.as_tensor(pred_matrix, dtype=torch.float32, device=torch_device)
    n_rows, n_models = pred_t.shape
    order = torch.argsort(pred_t, dim=0, stable=True)
    ranks = torch.empty_like(pred_t)
    rank_values = torch.arange(1, n_rows + 1, dtype=torch.float32, device=torch_device).unsqueeze(1)
    rank_values = rank_values.expand(-1, n_models)
    ranks.scatter_(0, order, rank_values)
    ranks = ranks / (n_rows + 1.0)
    w_t = torch.as_tensor(normalize_weights(weights), dtype=torch.float32, device=torch_device)
    return (ranks @ w_t).detach().cpu().numpy().astype(np.float64)


def rank_matrix(pred_matrix: np.ndarray) -> np.ndarray:
    n_rows = pred_matrix.shape[0]
    ranks = np.empty_like(pred_matrix, dtype=np.float64)
    for col in range(pred_matrix.shape[1]):
        order = np.argsort(pred_matrix[:, col], kind="mergesort")
        rank_values = np.empty(n_rows, dtype=np.float64)
        rank_values[order] = np.arange(1, n_rows + 1, dtype=np.float64)
        ranks[:, col] = rank_values
    return ranks


def median_blend(pred_matrix: np.ndarray) -> np.ndarray:
    return np.median(pred_matrix, axis=1).astype(np.float64)


def median_calibrated_blend(pred_matrix: np.ndarray) -> np.ndarray:
    ranks = rank_matrix(pred_matrix)
    median_rank = np.median(ranks, axis=1)
    median_pred = np.median(pred_matrix, axis=1)

    frame = pd.DataFrame({"median_rank": median_rank, "median_pred": median_pred})
    grouped = frame.groupby("median_rank", sort=True)["median_pred"].mean()
    calibrated_values = np.sort(grouped.to_numpy(dtype=np.float64))
    if len(calibrated_values) > 1:
        for i in range(1, len(calibrated_values)):
            if calibrated_values[i] <= calibrated_values[i - 1]:
                calibrated_values[i] = calibrated_values[i - 1] + 1e-9
    rank_to_pred = {float(r): float(v) for r, v in zip(grouped.index.to_numpy(), calibrated_values)}
    return np.fromiter((rank_to_pred[float(r)] for r in median_rank), dtype=np.float64, count=len(median_rank))


def order_adjusted_blend(
    pred_matrix: np.ndarray,
    base_weights: np.ndarray,
    rank_adjustments: np.ndarray,
    asc_weight: float,
    desc_weight: float,
) -> np.ndarray:
    if pred_matrix.shape[1] != len(base_weights):
        raise ValueError("Base weights length does not match number of submissions")
    if pred_matrix.shape[1] != len(rank_adjustments):
        raise ValueError("Rank-adjustments length must match number of submissions")

    mix_total = asc_weight + desc_weight
    if mix_total <= 0.0:
        raise ValueError("asc-weight + desc-weight must be positive")
    asc_ratio = asc_weight / mix_total
    desc_ratio = desc_weight / mix_total

    n_rows, n_models = pred_matrix.shape
    out = np.empty(n_rows, dtype=np.float64)

    for i in range(n_rows):
        row = pred_matrix[i]

        asc_order = np.argsort(row, kind="mergesort")
        asc_pos = np.empty(n_models, dtype=np.int32)
        asc_pos[asc_order] = np.arange(n_models)
        asc_pred = float(np.dot(row, base_weights + rank_adjustments[asc_pos]))

        desc_order = asc_order[::-1]
        desc_pos = np.empty(n_models, dtype=np.int32)
        desc_pos[desc_order] = np.arange(n_models)
        desc_pred = float(np.dot(row, base_weights + rank_adjustments[desc_pos]))

        out[i] = asc_ratio * asc_pred + desc_ratio * desc_pred

    return out


def order_adjusted_blend_torch(
    pred_matrix: np.ndarray,
    base_weights: np.ndarray,
    rank_adjustments: np.ndarray,
    asc_weight: float,
    desc_weight: float,
    torch_device: str,
) -> np.ndarray:
    import torch  # type: ignore

    if pred_matrix.shape[1] != len(base_weights):
        raise ValueError("Base weights length does not match number of submissions")
    if pred_matrix.shape[1] != len(rank_adjustments):
        raise ValueError("Rank-adjustments length must match number of submissions")

    mix_total = asc_weight + desc_weight
    if mix_total <= 0.0:
        raise ValueError("asc-weight + desc-weight must be positive")
    asc_ratio = asc_weight / mix_total
    desc_ratio = desc_weight / mix_total

    pred_t = torch.as_tensor(pred_matrix, dtype=torch.float32, device=torch_device)
    n_rows, n_models = pred_t.shape

    base_w_t = torch.as_tensor(base_weights, dtype=torch.float32, device=torch_device).unsqueeze(0)
    rank_adj_t = torch.as_tensor(rank_adjustments, dtype=torch.float32, device=torch_device)

    asc_order = torch.argsort(pred_t, dim=1, stable=True)
    order_positions = torch.arange(n_models, dtype=torch.int64, device=torch_device).unsqueeze(0)
    order_positions = order_positions.expand(n_rows, -1)

    asc_pos = torch.empty_like(asc_order)
    asc_pos.scatter_(1, asc_order, order_positions)
    asc_w = base_w_t + rank_adj_t[asc_pos]
    asc_pred = (pred_t * asc_w).sum(dim=1)

    desc_order = torch.flip(asc_order, dims=[1])
    desc_pos = torch.empty_like(desc_order)
    desc_pos.scatter_(1, desc_order, order_positions)
    desc_w = base_w_t + rank_adj_t[desc_pos]
    desc_pred = (pred_t * desc_w).sum(dim=1)

    out = asc_ratio * asc_pred + desc_ratio * desc_pred
    return out.detach().cpu().numpy().astype(np.float64)


def pairwise_metrics(pred_matrix: np.ndarray, names: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = pred_matrix[:, i]
            b = pred_matrix[:, j]
            rows.append(
                {
                    "left": names[i],
                    "right": names[j],
                    "mean_abs_diff": float(np.mean(np.abs(a - b))),
                    "pearson_corr": float(np.corrcoef(a, b)[0, 1]),
                }
            )
    return sorted(rows, key=lambda row: row["mean_abs_diff"], reverse=True)


def greedy_diverse_selection(
    pred_matrix: np.ndarray,
    model_names: list[str],
    max_models: int,
    min_corr: float,
) -> tuple[list[int], list[dict[str, float | str | None]]]:
    n_models = pred_matrix.shape[1]
    if max_models <= 0 or n_models <= max_models:
        return list(range(n_models)), []

    corr = np.corrcoef(pred_matrix.T)
    selected = [0]
    remaining = set(range(1, n_models))
    log: list[dict[str, float | str | None]] = [
        {"idx": 0, "file": model_names[0], "max_corr": None, "reason": "seed_first_input_anchor"}
    ]

    while len(selected) < max_models and remaining:
        best_idx: int | None = None
        best_max_corr = float("inf")
        for idx in sorted(remaining):
            max_corr = max(float(abs(corr[idx, s])) for s in selected)
            if max_corr < best_max_corr:
                best_max_corr = max_corr
                best_idx = idx

        if best_idx is None:
            break
        if best_max_corr > min_corr:
            log.append(
                {
                    "idx": None,
                    "file": None,
                    "max_corr": float(best_max_corr),
                    "reason": f"stopped_all_remaining_corr_above_threshold_{min_corr}",
                }
            )
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        log.append(
            {
                "idx": int(best_idx),
                "file": model_names[best_idx],
                "max_corr": float(best_max_corr),
                "reason": "selected_lowest_max_corr",
            }
        )

    return selected, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend multiple Kaggle submission CSV files with weighted/rank/order-adjusted methods."
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated submission file paths (example: outputs/a.csv,outputs/b.csv)",
    )
    parser.add_argument("--output", required=True, help="Output blended submission CSV")
    parser.add_argument("--report", default="", help="Optional diagnostics JSON output path")
    parser.add_argument("--id-column", default="id", help="Submission id column")
    parser.add_argument("--target-column", default="Churn", help="Submission prediction column")
    parser.add_argument(
        "--method",
        default="order",
        choices=["weighted", "rank", "order", "median", "median_calibrated"],
        help="Blending method",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Base model weights, comma-separated. Defaults to equal weights.",
    )
    parser.add_argument(
        "--rank-adjustments",
        default="",
        help="Order-position adjustments for order method (length = number of models). Defaults to zeros.",
    )
    parser.add_argument("--asc-weight", type=float, default=0.30, help="Asc blend mix weight for order method")
    parser.add_argument("--desc-weight", type=float, default=0.70, help="Desc blend mix weight for order method")
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable clipping output probabilities to [0, 1]",
    )
    parser.add_argument(
        "--compute-device",
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Compute backend for blending math (auto tries CUDA via torch, then CPU numpy)",
    )
    parser.add_argument(
        "--diverse-max-models",
        type=int,
        default=0,
        help="Greedy correlation-based pre-selection cap (0 disables, first input stays anchor seed)",
    )
    parser.add_argument(
        "--diverse-min-corr",
        type=float,
        default=0.998,
        help="Stop adding models when best candidate max correlation exceeds this threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in parse_list(args.inputs)]

    merged, pred_cols = load_submissions(input_paths, args.id_column, args.target_column)
    pred_matrix_full = merged[pred_cols].to_numpy(dtype=np.float64)
    model_names_full = [path.name for path in input_paths]
    selected_indices, selection_log = greedy_diverse_selection(
        pred_matrix_full,
        model_names_full,
        args.diverse_max_models,
        args.diverse_min_corr,
    )

    if len(selected_indices) < len(model_names_full):
        print(
            f"Diversity selection enabled: using {len(selected_indices)}/{len(model_names_full)} models "
            f"(threshold={args.diverse_min_corr})"
        )
    pred_matrix = pred_matrix_full[:, selected_indices]
    model_names = [model_names_full[i] for i in selected_indices]
    selected_paths = [input_paths[i] for i in selected_indices]
    n_models = pred_matrix.shape[1]

    torch_device, compute_note = resolve_compute_backend(args.compute_device)
    print(f"Compute backend: {compute_note}")

    if args.weights:
        parsed_weights = np.asarray(parse_float_list(args.weights), dtype=np.float64)
        if len(parsed_weights) != len(model_names_full):
            if len(parsed_weights) != n_models:
                raise ValueError(
                    f"weights length {len(parsed_weights)} must equal original model count "
                    f"{len(model_names_full)} or selected model count {n_models}"
                )
            base_weights = parsed_weights
        else:
            base_weights = parsed_weights[selected_indices]
    else:
        base_weights = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    if len(base_weights) != n_models:
        raise ValueError(f"weights length {len(base_weights)} != number of models {n_models}")

    if args.method == "weighted":
        if torch_device is not None:
            blend = weighted_average_torch(pred_matrix, base_weights, torch_device)
        else:
            blend = weighted_average(pred_matrix, base_weights)
        rank_adjustments = np.zeros(n_models, dtype=np.float64)
    elif args.method == "rank":
        if torch_device is not None:
            blend = weighted_rank_average_torch(pred_matrix, base_weights, torch_device)
        else:
            blend = weighted_rank_average(pred_matrix, base_weights)
        rank_adjustments = np.zeros(n_models, dtype=np.float64)
    elif args.method == "median":
        blend = median_blend(pred_matrix)
        rank_adjustments = np.zeros(n_models, dtype=np.float64)
    elif args.method == "median_calibrated":
        blend = median_calibrated_blend(pred_matrix)
        rank_adjustments = np.zeros(n_models, dtype=np.float64)
    else:
        if args.rank_adjustments:
            parsed_rank_adjustments = np.asarray(parse_float_list(args.rank_adjustments), dtype=np.float64)
            if len(parsed_rank_adjustments) == len(model_names_full):
                rank_adjustments = parsed_rank_adjustments[selected_indices]
            elif len(parsed_rank_adjustments) == n_models:
                rank_adjustments = parsed_rank_adjustments
            else:
                raise ValueError(
                    f"rank-adjustments length {len(parsed_rank_adjustments)} must equal original model count "
                    f"{len(model_names_full)} or selected model count {n_models}"
                )
        else:
            rank_adjustments = np.zeros(n_models, dtype=np.float64)
        if len(rank_adjustments) != n_models:
            raise ValueError(
                f"rank-adjustments length {len(rank_adjustments)} != number of models {n_models}"
            )
        if torch_device is not None:
            blend = order_adjusted_blend_torch(
                pred_matrix=pred_matrix,
                base_weights=base_weights,
                rank_adjustments=rank_adjustments,
                asc_weight=args.asc_weight,
                desc_weight=args.desc_weight,
                torch_device=torch_device,
            )
        else:
            blend = order_adjusted_blend(
                pred_matrix=pred_matrix,
                base_weights=base_weights,
                rank_adjustments=rank_adjustments,
                asc_weight=args.asc_weight,
                desc_weight=args.desc_weight,
            )

    if not args.no_clip:
        blend = np.clip(blend, 0.0, 1.0)

    out_df = pd.DataFrame({args.id_column: merged[args.id_column], args.target_column: blend})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Blended {n_models} files with method='{args.method}'")
    print(f"Output written to: {out_path}")
    print(
        f"Prediction stats: min={blend.min():.6f}, max={blend.max():.6f}, "
        f"mean={blend.mean():.6f}, std={blend.std():.6f}"
    )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pairwise = pairwise_metrics(pred_matrix, model_names)
        report = {
            "method": args.method,
            "input_files": [str(p) for p in input_paths],
            "selected_files": [str(p) for p in selected_paths],
            "selected_indices": selected_indices,
            "selection_log": selection_log,
            "diverse_max_models": args.diverse_max_models,
            "diverse_min_corr": args.diverse_min_corr,
            "compute_device_requested": args.compute_device,
            "compute_backend_resolved": compute_note,
            "weights": base_weights.tolist(),
            "rank_adjustments": rank_adjustments.tolist(),
            "asc_weight": args.asc_weight,
            "desc_weight": args.desc_weight,
            "prediction_summary": {
                "min": float(blend.min()),
                "max": float(blend.max()),
                "mean": float(blend.mean()),
                "std": float(blend.std()),
            },
            "pairwise_top_diverse": pairwise[: min(20, len(pairwise))],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Diagnostics report written to: {report_path}")


if __name__ == "__main__":
    main()
