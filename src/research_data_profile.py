import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TARGET_COL = "Churn"
ID_COL = "id"

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
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

INTERNET_DEPENDENT_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full EDA + feature-research report for tabular churn data.")
    parser.add_argument("--train", default="data/train.csv", help="Path to train CSV")
    parser.add_argument("--test", default="data/test.csv", help="Path to test CSV")
    parser.add_argument(
        "--out-json",
        default="outputs/data_research_report.json",
        help="Output JSON report",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/data_research_report.md",
        help="Output markdown report",
    )
    return parser.parse_args()


def as_float(text: str) -> float:
    return float(text)


def as_int(text: str) -> int:
    return int(text)


def quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarize_numeric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    n = len(values)
    v_sorted = sorted(values)
    mean = sum(values) / n
    var = sum((x - mean) * (x - mean) for x in values) / n
    std = math.sqrt(var)
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": v_sorted[0],
        "p01": quantile(v_sorted, 0.01),
        "p05": quantile(v_sorted, 0.05),
        "p25": quantile(v_sorted, 0.25),
        "p50": quantile(v_sorted, 0.50),
        "p75": quantile(v_sorted, 0.75),
        "p95": quantile(v_sorted, 0.95),
        "p99": quantile(v_sorted, 0.99),
        "max": v_sorted[-1],
    }


def format_num(x: float) -> str:
    if x != x:  # NaN check
        return "nan"
    return f"{x:.6f}"


def rate(pos: int, total: int) -> float:
    return float(pos) / float(total) if total else 0.0


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def tv_distance(counter_a: Counter, n_a: int, counter_b: Counter, n_b: int) -> float:
    keys = set(counter_a) | set(counter_b)
    return 0.5 * sum(abs(counter_a.get(k, 0) / n_a - counter_b.get(k, 0) / n_b) for k in keys)


def psi_numeric(train_values: List[float], test_values: List[float], bins: int = 10, eps: float = 1e-6) -> float:
    if not train_values or not test_values:
        return 0.0
    t_sorted = sorted(train_values)
    edges = [quantile(t_sorted, i / bins) for i in range(bins + 1)]

    # Deduplicate edges to avoid zero-width bins.
    dedup = [edges[0]]
    for e in edges[1:]:
        if e > dedup[-1]:
            dedup.append(e)
    if len(dedup) <= 2:
        return 0.0
    edges = dedup

    def bin_counts(values: List[float]) -> List[int]:
        counts = [0] * (len(edges) - 1)
        for v in values:
            idx = len(edges) - 2
            for i in range(len(edges) - 1):
                left = edges[i]
                right = edges[i + 1]
                if i == len(edges) - 2:
                    if left <= v <= right:
                        idx = i
                        break
                else:
                    if left <= v < right:
                        idx = i
                        break
            counts[idx] += 1
        return counts

    ct_train = bin_counts(train_values)
    ct_test = bin_counts(test_values)
    n_train = len(train_values)
    n_test = len(test_values)

    psi = 0.0
    for a, b in zip(ct_train, ct_test):
        p = max(a / n_train, eps)
        q = max(b / n_test, eps)
        psi += (p - q) * math.log(p / q)
    return psi


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_train_profile(rows: List[Dict[str, str]]) -> Dict[str, object]:
    n_rows = len(rows)
    missing = Counter()
    id_seen = set()
    id_duplicates = 0

    target_counter = Counter()
    numeric_values = {c: [] for c in NUMERIC_COLS}
    numeric_pos = {c: [] for c in NUMERIC_COLS}
    numeric_neg = {c: [] for c in NUMERIC_COLS}
    cat_counter = {c: Counter() for c in CATEGORICAL_COLS}
    cat_target = {c: defaultdict(lambda: [0, 0]) for c in CATEGORICAL_COLS}  # value -> [neg, pos]

    tenure_bins = defaultdict(lambda: [0, 0])
    monthly_bins = defaultdict(lambda: [0, 0])
    contract_tenure_bins = defaultdict(lambda: [0, 0])
    payment_paper_bins = defaultdict(lambda: [0, 0])
    internet_services_yes_bins = defaultdict(lambda: [0, 0])

    internet_logic_violations = 0
    phone_logic_violations = 0
    total_gap_values = []
    total_gap_abs_values = []

    for row in rows:
        for k, v in row.items():
            if v == "":
                missing[k] += 1

        rid = row[ID_COL]
        if rid in id_seen:
            id_duplicates += 1
        id_seen.add(rid)

        y = 1 if row[TARGET_COL] == "Yes" else 0
        target_counter[row[TARGET_COL]] += 1

        for c in NUMERIC_COLS:
            if c in ("SeniorCitizen", "tenure"):
                val = as_int(row[c])
            else:
                val = as_float(row[c])
            numeric_values[c].append(float(val))
            if y == 1:
                numeric_pos[c].append(float(val))
            else:
                numeric_neg[c].append(float(val))

        for c in CATEGORICAL_COLS:
            val = row[c]
            cat_counter[c][val] += 1
            cat_target[c][val][y] += 1

        tenure = as_int(row["tenure"])
        monthly = as_float(row["MonthlyCharges"])
        total = as_float(row["TotalCharges"])
        gap = total - (tenure * monthly)
        total_gap_values.append(gap)
        total_gap_abs_values.append(abs(gap))

        if tenure <= 6:
            t_bin = "0-6"
        elif tenure <= 12:
            t_bin = "7-12"
        elif tenure <= 24:
            t_bin = "13-24"
        elif tenure <= 36:
            t_bin = "25-36"
        elif tenure <= 48:
            t_bin = "37-48"
        elif tenure <= 60:
            t_bin = "49-60"
        else:
            t_bin = "61-72"
        tenure_bins[t_bin][y] += 1
        contract_tenure_bins[(row["Contract"], t_bin)][y] += 1

        if monthly < 35:
            m_bin = "low(<35)"
        elif monthly < 70:
            m_bin = "mid(35-70)"
        elif monthly < 90:
            m_bin = "high(70-90)"
        else:
            m_bin = "very_high(>=90)"
        monthly_bins[m_bin][y] += 1

        payment_paper_bins[(row["PaymentMethod"], row["PaperlessBilling"])][y] += 1

        services_yes = 0
        for col in INTERNET_DEPENDENT_COLS:
            if row[col] == "Yes":
                services_yes += 1
        internet_services_yes_bins[(row["InternetService"], services_yes)][y] += 1

        if row["InternetService"] == "No":
            for col in INTERNET_DEPENDENT_COLS:
                if row[col] != "No internet service":
                    internet_logic_violations += 1
                    break
        if row["PhoneService"] == "No" and row["MultipleLines"] != "No phone service":
            phone_logic_violations += 1

    pos_total = target_counter.get("Yes", 0)
    neg_total = target_counter.get("No", 0)
    churn_rate = rate(pos_total, n_rows)

    numeric_summary = {}
    for c in NUMERIC_COLS:
        numeric_summary[c] = {
            "overall": summarize_numeric(numeric_values[c]),
            "churn_yes": summarize_numeric(numeric_pos[c]),
            "churn_no": summarize_numeric(numeric_neg[c]),
        }

    categorical_summary = {}
    for c in CATEGORICAL_COLS:
        per_value = []
        for val, cnt in cat_counter[c].most_common():
            neg, pos = cat_target[c][val][0], cat_target[c][val][1]
            val_rate = rate(pos, cnt)
            per_value.append(
                {
                    "value": val,
                    "count": cnt,
                    "share": rate(cnt, n_rows),
                    "churn_rate": val_rate,
                    "lift_vs_global": safe_div(val_rate, churn_rate) if churn_rate else 0.0,
                }
            )
        categorical_summary[c] = {
            "unique_count": len(cat_counter[c]),
            "top_values": per_value[: min(20, len(per_value))],
        }

    def sorted_group_rates(group: Dict[object, List[int]], min_count: int = 5000) -> List[Dict[str, object]]:
        rows_out = []
        for k, arr in group.items():
            neg, pos = arr[0], arr[1]
            total = neg + pos
            if total < min_count:
                continue
            rows_out.append(
                {
                    "key": str(k),
                    "count": total,
                    "churn_rate": rate(pos, total),
                    "lift_vs_global": safe_div(rate(pos, total), churn_rate) if churn_rate else 0.0,
                }
            )
        rows_out.sort(key=lambda r: (r["churn_rate"], r["count"]), reverse=True)
        return rows_out

    profile = {
        "rows": n_rows,
        "id_unique": len(id_seen),
        "id_duplicates": id_duplicates,
        "missing_counts": dict(missing),
        "target_counts": dict(target_counter),
        "target_churn_rate": churn_rate,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "derived_checks": {
            "internet_logic_violations": internet_logic_violations,
            "phone_logic_violations": phone_logic_violations,
            "total_gap_summary": summarize_numeric(total_gap_values),
            "abs_total_gap_summary": summarize_numeric(total_gap_abs_values),
        },
        "group_risks": {
            "tenure_bins": sorted_group_rates(tenure_bins, min_count=2000),
            "monthly_bins": sorted_group_rates(monthly_bins, min_count=2000),
            "contract_x_tenure_bin": sorted_group_rates(contract_tenure_bins, min_count=2000),
            "payment_x_paperless": sorted_group_rates(payment_paper_bins, min_count=2000),
            "internetservice_x_services_yes": sorted_group_rates(internet_services_yes_bins, min_count=2000),
        },
    }
    return profile


def compute_test_profile(rows: List[Dict[str, str]]) -> Dict[str, object]:
    n_rows = len(rows)
    missing = Counter()
    id_seen = set()
    id_duplicates = 0
    numeric_values = {c: [] for c in NUMERIC_COLS}
    cat_counter = {c: Counter() for c in CATEGORICAL_COLS}

    internet_logic_violations = 0
    phone_logic_violations = 0
    total_gap_values = []
    total_gap_abs_values = []

    for row in rows:
        for k, v in row.items():
            if v == "":
                missing[k] += 1

        rid = row[ID_COL]
        if rid in id_seen:
            id_duplicates += 1
        id_seen.add(rid)

        for c in NUMERIC_COLS:
            if c in ("SeniorCitizen", "tenure"):
                val = as_int(row[c])
            else:
                val = as_float(row[c])
            numeric_values[c].append(float(val))

        for c in CATEGORICAL_COLS:
            val = row[c]
            cat_counter[c][val] += 1

        tenure = as_int(row["tenure"])
        monthly = as_float(row["MonthlyCharges"])
        total = as_float(row["TotalCharges"])
        gap = total - (tenure * monthly)
        total_gap_values.append(gap)
        total_gap_abs_values.append(abs(gap))

        if row["InternetService"] == "No":
            for col in INTERNET_DEPENDENT_COLS:
                if row[col] != "No internet service":
                    internet_logic_violations += 1
                    break
        if row["PhoneService"] == "No" and row["MultipleLines"] != "No phone service":
            phone_logic_violations += 1

    profile = {
        "rows": n_rows,
        "id_unique": len(id_seen),
        "id_duplicates": id_duplicates,
        "missing_counts": dict(missing),
        "numeric_summary": {c: summarize_numeric(v) for c, v in numeric_values.items()},
        "categorical_counts": {c: dict(v) for c, v in cat_counter.items()},
        "derived_checks": {
            "internet_logic_violations": internet_logic_violations,
            "phone_logic_violations": phone_logic_violations,
            "total_gap_summary": summarize_numeric(total_gap_values),
            "abs_total_gap_summary": summarize_numeric(total_gap_abs_values),
        },
    }
    return profile


def compute_shift(
    train_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]],
) -> Dict[str, object]:
    n_train = len(train_rows)
    n_test = len(test_rows)

    train_num = {c: [] for c in NUMERIC_COLS}
    test_num = {c: [] for c in NUMERIC_COLS}
    train_cat = {c: Counter() for c in CATEGORICAL_COLS}
    test_cat = {c: Counter() for c in CATEGORICAL_COLS}

    for row in train_rows:
        for c in NUMERIC_COLS:
            if c in ("SeniorCitizen", "tenure"):
                train_num[c].append(float(as_int(row[c])))
            else:
                train_num[c].append(as_float(row[c]))
        for c in CATEGORICAL_COLS:
            train_cat[c][row[c]] += 1

    for row in test_rows:
        for c in NUMERIC_COLS:
            if c in ("SeniorCitizen", "tenure"):
                test_num[c].append(float(as_int(row[c])))
            else:
                test_num[c].append(as_float(row[c]))
        for c in CATEGORICAL_COLS:
            test_cat[c][row[c]] += 1

    numeric_shift = {}
    for c in NUMERIC_COLS:
        s_tr = summarize_numeric(train_num[c])
        s_te = summarize_numeric(test_num[c])
        numeric_shift[c] = {
            "train_mean": s_tr["mean"],
            "test_mean": s_te["mean"],
            "mean_diff": s_te["mean"] - s_tr["mean"],
            "train_std": s_tr["std"],
            "test_std": s_te["std"],
            "psi": psi_numeric(train_num[c], test_num[c], bins=10),
        }

    categorical_shift = {}
    for c in CATEGORICAL_COLS:
        categorical_shift[c] = {
            "tv_distance": tv_distance(train_cat[c], n_train, test_cat[c], n_test),
            "train_unique": len(train_cat[c]),
            "test_unique": len(test_cat[c]),
        }

    return {"numeric_shift": numeric_shift, "categorical_shift": categorical_shift}


def top_feature_ideas(profile_train: Dict[str, object]) -> List[Dict[str, object]]:
    churn_rate = float(profile_train["target_churn_rate"])
    cat_sum = profile_train["categorical_summary"]

    ideas: List[Dict[str, object]] = []

    def best_lift(feature: str) -> Tuple[str, float, float, int]:
        best_value = ""
        best_lift_val = 0.0
        best_rate = 0.0
        best_count = 0
        for row in cat_sum[feature]["top_values"]:
            if row["count"] < 3000:
                continue
            if row["lift_vs_global"] > best_lift_val:
                best_value = row["value"]
                best_lift_val = row["lift_vs_global"]
                best_rate = row["churn_rate"]
                best_count = row["count"]
        return best_value, best_lift_val, best_rate, best_count

    # 1) Payment method risk split.
    val, lift, r, cnt = best_lift("PaymentMethod")
    ideas.append(
        {
            "feature_name": "is_electronic_check",
            "type": "binary",
            "why": f"PaymentMethod='{val}' has churn_rate={r:.4f} (lift={lift:.2f}, n={cnt}), global={churn_rate:.4f}",
            "formula": "1 if PaymentMethod == 'Electronic check' else 0",
        }
    )

    # 2) Contract-risk ordering.
    contract_rows = cat_sum["Contract"]["top_values"]
    ideas.append(
        {
            "feature_name": "contract_term_ordinal",
            "type": "ordinal",
            "why": "Contract shows strong risk stratification across Month-to-month / One year / Two year.",
            "formula": "map {'Month-to-month':0, 'One year':1, 'Two year':2}",
        }
    )

    # 3) Services yes count already exists, add squared/nonlinear buckets.
    ideas.append(
        {
            "feature_name": "internet_services_yes_bucket",
            "type": "categorical",
            "why": "Nonlinear churn response expected by count of subscribed internet add-on services.",
            "formula": "bucket internet_services_yes into {0,1-2,3-4,5-6}",
        }
    )

    # 4) Billing friction.
    ideas.append(
        {
            "feature_name": "autopay_flag",
            "type": "binary",
            "why": "Automatic payment methods generally reduce churn risk vs manual/electronic check.",
            "formula": "1 if PaymentMethod contains '(automatic)' else 0",
        }
    )

    # 5) Price pressure.
    ideas.append(
        {
            "feature_name": "monthly_charge_bin_x_contract",
            "type": "categorical_interaction",
            "why": "High monthly charges combined with month-to-month contract often indicate higher churn.",
            "formula": "concat(monthly_charge_bin, Contract)",
        }
    )

    # 6) Coherence and data generation artifact proxy.
    ideas.append(
        {
            "feature_name": "charges_gap_abs",
            "type": "numeric",
            "why": "Absolute gap |TotalCharges - tenure*MonthlyCharges| may capture billing pattern effects.",
            "formula": "abs(TotalCharges - tenure * MonthlyCharges)",
        }
    )

    # 7) Tenure lifecycle curve.
    ideas.append(
        {
            "feature_name": "tenure_bin",
            "type": "categorical",
            "why": "Churn risk is usually highest early and decreases with longer tenure.",
            "formula": "bucket tenure into [0-6,7-12,13-24,25-36,37-48,49-60,61-72]",
        }
    )

    # 8) Family stability interaction.
    ideas.append(
        {
            "feature_name": "household_stability",
            "type": "categorical_interaction",
            "why": "Partner/Dependents combination can interact with tenure and contract risk.",
            "formula": "concat(Partner, Dependents, Contract)",
        }
    )

    return ideas


def write_markdown(report: Dict[str, object], out_path: Path) -> None:
    train = report["train_profile"]
    test = report["test_profile"]
    shift = report["shift_analysis"]
    ideas = report["feature_ideas"]

    lines: List[str] = []
    lines.append("# Data Research Report - Predict Customer Churn")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- Train rows: {train['rows']}")
    lines.append(f"- Test rows: {test['rows']}")
    lines.append(f"- Train churn rate: {format_num(train['target_churn_rate'])}")
    lines.append(f"- Train id duplicates: {train['id_duplicates']}")
    lines.append(f"- Test id duplicates: {test['id_duplicates']}")
    lines.append("")

    lines.append("## Missingness / Data Quality")
    lines.append(f"- Train missing fields total: {sum(train['missing_counts'].values())}")
    lines.append(f"- Test missing fields total: {sum(test['missing_counts'].values())}")
    lines.append(
        f"- Internet-service logic violations (train/test): "
        f"{train['derived_checks']['internet_logic_violations']} / {test['derived_checks']['internet_logic_violations']}"
    )
    lines.append(
        f"- Phone-service logic violations (train/test): "
        f"{train['derived_checks']['phone_logic_violations']} / {test['derived_checks']['phone_logic_violations']}"
    )
    lines.append("")

    lines.append("## Numeric Features (Train)")
    for c, stats in train["numeric_summary"].items():
        ov = stats["overall"]
        yes = stats["churn_yes"]
        no = stats["churn_no"]
        lines.append(
            f"- {c}: mean={format_num(ov['mean'])}, std={format_num(ov['std'])}, "
            f"p50={format_num(ov['p50'])}, p95={format_num(ov['p95'])}"
        )
        lines.append(
            f"  churn_yes_mean={format_num(yes['mean'])}, churn_no_mean={format_num(no['mean'])}, "
            f"delta={format_num(yes['mean'] - no['mean'])}"
        )
    lines.append("")

    lines.append("## Highest-Risk Groups (Train)")
    for group_name, rows in train["group_risks"].items():
        lines.append(f"- {group_name}:")
        for row in rows[:5]:
            lines.append(
                f"  - {row['key']} | n={row['count']} | churn_rate={format_num(row['churn_rate'])} "
                f"| lift={format_num(row['lift_vs_global'])}"
            )
    lines.append("")

    lines.append("## Train/Test Shift")
    lines.append("- Numeric shift (mean_diff, PSI):")
    for c, s in shift["numeric_shift"].items():
        lines.append(
            f"  - {c}: mean_diff={format_num(s['mean_diff'])}, psi={format_num(s['psi'])}, "
            f"train_mean={format_num(s['train_mean'])}, test_mean={format_num(s['test_mean'])}"
        )
    lines.append("- Categorical shift (TV distance):")
    cat_sorted = sorted(
        shift["categorical_shift"].items(),
        key=lambda kv: kv[1]["tv_distance"],
        reverse=True,
    )
    for c, s in cat_sorted:
        lines.append(f"  - {c}: tv_distance={format_num(s['tv_distance'])}")
    lines.append("")

    lines.append("## Feature Engineering Recommendations")
    for idx, idea in enumerate(ideas, start=1):
        lines.append(f"{idx}. `{idea['feature_name']}` ({idea['type']})")
        lines.append(f"   - Why: {idea['why']}")
        lines.append(f"   - Formula: `{idea['formula']}`")
    lines.append("")

    lines.append("## Notes")
    lines.append(
        "- This report is generated with pure-Python CSV processing so it can run even without pandas/numpy."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    train_rows = read_csv_rows(train_path)
    test_rows = read_csv_rows(test_path)

    train_profile = compute_train_profile(train_rows)
    test_profile = compute_test_profile(test_rows)
    shift = compute_shift(train_rows, test_rows)
    ideas = top_feature_ideas(train_profile)

    report = {
        "train_profile": train_profile,
        "test_profile": test_profile,
        "shift_analysis": shift,
        "feature_ideas": ideas,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_md)

    print(f"Train rows: {train_profile['rows']}, Test rows: {test_profile['rows']}")
    print(f"Train churn rate: {train_profile['target_churn_rate']:.6f}")
    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")


if __name__ == "__main__":
    main()
