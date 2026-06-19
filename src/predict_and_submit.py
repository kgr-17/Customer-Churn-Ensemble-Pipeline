import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = ["id", "Churn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline training and optional Kaggle submission")
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument("--submission", default="outputs/submission_baseline.csv", help="Output submission CSV")
    parser.add_argument("--metrics", default="outputs/metrics_baseline.json", help="Output metrics JSON")
    parser.add_argument("--model", default="artifacts/catboost_baseline.cbm", help="Output model path")
    parser.add_argument("--competition", default=os.getenv("KAGGLE_COMPETITION", ""), help="Kaggle competition slug")
    parser.add_argument("--message", default="catboost baseline v1", help="Submission message")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only validate/submit")
    parser.add_argument("--dry-run", action="store_true", help="Validate Kaggle auth and competition access")
    parser.add_argument("--submit", action="store_true", help="Submit the CSV to Kaggle")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "src/train_baseline.py",
        "--train",
        args.train,
        "--test",
        args.test,
        "--submission",
        args.submission,
        "--metrics",
        args.metrics,
        "--model",
        args.model,
    ]
    subprocess.run(cmd, check=True)


def validate_submission(path: str) -> int:
    submission_path = Path(path)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    df = pd.read_csv(submission_path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {EXPECTED_COLUMNS}, got {list(df.columns)}")

    if df["Churn"].isna().any():
        raise ValueError("Submission contains NaN values in Churn")

    if ((df["Churn"] < 0.0) | (df["Churn"] > 1.0)).any():
        raise ValueError("Submission probabilities must be within [0, 1]")

    return len(df)


def kaggle_preflight(competition: str) -> None:
    subprocess.run(["kaggle", "competitions", "files", "-c", competition], check=True)


def kaggle_submit(competition: str, submission: str, message: str) -> None:
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition,
            "-f",
            submission,
            "-m",
            message,
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()

    if not args.skip_train:
        run_training(args)

    row_count = validate_submission(args.submission)
    print(f"Submission schema check passed ({row_count} rows)")

    if args.submit or args.dry_run:
        if not args.competition:
            raise ValueError("KAGGLE_COMPETITION is required for --dry-run or --submit")
        kaggle_preflight(args.competition)
        print("Kaggle preflight passed")

    if args.submit:
        kaggle_submit(args.competition, args.submission, args.message)
        print("Kaggle submission completed")


if __name__ == "__main__":
    main()
