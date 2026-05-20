from __future__ import annotations

import argparse
from pathlib import Path

from build_sqlite import build_database
from churn_data import clean_telco_data, engineer_features, load_telco_data, save_processed_data
from churn_modeling import run_modeling_pipeline
from churn_visuals import generate_visuals
from paths import ensure_project_dirs


def prepare_data(raw_csv: Path | str | None = None) -> Path:
    raw_df = load_telco_data(raw_csv)
    processed_df = engineer_features(clean_telco_data(raw_df))
    return save_processed_data(processed_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Customer Churn Prediction and Retention Analysis pipeline."
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Optional path to WA_Fn-UseC_-Telco-Customer-Churn.csv.",
    )
    parser.add_argument("--skip-visuals", action="store_true", help="Skip EDA chart generation.")
    parser.add_argument("--skip-models", action="store_true", help="Skip ML model training.")
    parser.add_argument("--skip-sqlite", action="store_true", help="Skip SQLite export.")
    parser.add_argument(
        "--no-xgboost",
        action="store_true",
        help="Skip optional XGBoost model even if the package is installed.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    processed_path = prepare_data(args.raw_csv)
    print(f"Processed dataset saved to {processed_path}")

    if not args.skip_visuals:
        visuals = generate_visuals(raw_csv=args.raw_csv)
        print(f"Generated {len(visuals)} EDA visuals.")

    if not args.skip_models:
        run_modeling_pipeline(raw_csv=args.raw_csv, include_xgboost=not args.no_xgboost)

    if not args.skip_sqlite:
        db_path = build_database(raw_csv=args.raw_csv)
        print(f"SQLite database saved to {db_path}")


if __name__ == "__main__":
    main()
