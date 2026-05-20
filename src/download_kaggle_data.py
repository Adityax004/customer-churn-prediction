from __future__ import annotations

import argparse

from paths import RAW_DATA_PATH, RAW_DIR, ensure_project_dirs


KAGGLE_DATASET = "blastchar/telco-customer-churn"


def download_telco_churn(force: bool = False) -> None:
    """Download the Telco Customer Churn CSV through the Kaggle API."""
    ensure_project_dirs()

    if RAW_DATA_PATH.exists() and not force:
        print(f"Raw dataset already exists: {RAW_DATA_PATH}")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Install the Kaggle package with `pip install kaggle` before downloading."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Download completed, but expected CSV was not found at {RAW_DATA_PATH}."
        )

    print(f"Downloaded dataset to {RAW_DATA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Telco Customer Churn from Kaggle.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the CSV exists.")
    args = parser.parse_args()
    download_telco_churn(force=args.force)


if __name__ == "__main__":
    main()
