from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from paths import PROCESSED_DATA_PATH, RAW_DATA_PATH, ensure_project_dirs


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

FEATURE_CATEGORIES = {
    "demographic": ["gender", "SeniorCitizen", "Partner", "Dependents"],
    "service_related": [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ],
    "account_related": [
        "tenure",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ],
    "target": [TARGET_COLUMN],
}

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: Any
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


def load_telco_data(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV."""
    path = Path(csv_path) if csv_path is not None else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Telco churn CSV not found. Place "
            "`WA_Fn-UseC_-Telco-Customer-Churn.csv` in data/raw/ or run "
            "`python src/download_kaggle_data.py` with Kaggle credentials."
        )
    return pd.read_csv(path)


def dataset_overview(df: pd.DataFrame) -> dict[str, object]:
    """Return quick metadata for notebook display and audits."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def blank_value_counts(df: pd.DataFrame) -> pd.Series:
    string_like = df.select_dtypes(include=["object", "string"]).columns
    return df[string_like].apply(lambda col: col.astype(str).str.strip().eq("").sum()).sort_values(
        ascending=False
    )


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize common quality issues in the Telco churn dataset."""
    report_rows = [
        {"check": "rows", "value": len(df), "notes": "Total records in the dataset."},
        {
            "check": "columns",
            "value": len(df.columns),
            "notes": "Expected raw dataset has 21 columns.",
        },
        {
            "check": "duplicate_rows",
            "value": int(df.duplicated().sum()),
            "notes": "Exact duplicate rows.",
        },
    ]

    if ID_COLUMN in df.columns:
        report_rows.append(
            {
                "check": "duplicate_customer_ids",
                "value": int(df[ID_COLUMN].duplicated().sum()),
                "notes": "Duplicate customer identifiers.",
            }
        )

    if "TotalCharges" in df.columns:
        total_charges = df["TotalCharges"].astype(str).str.strip()
        invalid_total_charges = pd.to_numeric(total_charges.replace("", np.nan), errors="coerce").isna()
        report_rows.append(
            {
                "check": "blank_total_charges",
                "value": int(total_charges.eq("").sum()),
                "notes": "Blank values usually correspond to zero-tenure accounts.",
            }
        )
        report_rows.append(
            {
                "check": "non_numeric_total_charges",
                "value": int(invalid_total_charges.sum()),
                "notes": "Rows that require numeric conversion or repair.",
            }
        )

    for column in ["tenure", "MonthlyCharges"]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            report_rows.append(
                {
                    "check": f"negative_{column}",
                    "value": int((values < 0).sum()),
                    "notes": f"Invalid negative values in {column}.",
                }
            )

    blanks = blank_value_counts(df)
    for column, blank_count in blanks[blanks > 0].items():
        report_rows.append(
            {
                "check": f"blank_values_{column}",
                "value": int(blank_count),
                "notes": "Blank or whitespace-only string values.",
            }
        )

    return pd.DataFrame(report_rows)


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    return cleaned


def _standardize_senior_citizen(series: pd.Series) -> pd.Series:
    mapped = series.replace({1: "Yes", 0: "No", "1": "Yes", "0": "No"})
    return mapped.astype("string").str.strip().fillna("Unknown")


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco churn data and repair known inconsistencies."""
    cleaned = strip_string_columns(df)

    if ID_COLUMN in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=[ID_COLUMN], keep="first")
    else:
        cleaned = cleaned.drop_duplicates(keep="first")

    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"].astype("string").str.strip().replace("", np.nan),
        errors="coerce",
    )
    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")

    zero_tenure_missing_total = cleaned["TotalCharges"].isna() & cleaned["tenure"].fillna(-1).eq(0)
    cleaned.loc[zero_tenure_missing_total, "TotalCharges"] = 0.0

    remaining_missing_total = cleaned["TotalCharges"].isna()
    cleaned.loc[remaining_missing_total, "TotalCharges"] = (
        cleaned.loc[remaining_missing_total, "MonthlyCharges"]
        * cleaned.loc[remaining_missing_total, "tenure"]
    )

    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = _standardize_senior_citizen(cleaned["SeniorCitizen"])

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("string").str.title()
        cleaned = cleaned[cleaned[TARGET_COLUMN].isin(["Yes", "No"])]

    cleaned = cleaned.dropna(subset=["tenure", "MonthlyCharges", "TotalCharges"])
    cleaned = cleaned[cleaned["tenure"].between(0, 72)]
    cleaned = cleaned[cleaned["MonthlyCharges"].ge(0)]
    cleaned = cleaned[cleaned["TotalCharges"].ge(0)]

    return cleaned.reset_index(drop=True)


def _count_active_services(row: pd.Series) -> int:
    count = 0
    if row.get("PhoneService") == "Yes":
        count += 1
    if row.get("InternetService") in {"DSL", "Fiber optic"}:
        count += 1

    for column in [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        if row.get(column) == "Yes":
            count += 1
    return count


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-oriented features for EDA and machine learning."""
    engineered = df.copy()

    tenure_bins = [-0.01, 12, 24, 48, 60, np.inf]
    tenure_labels = ["0-12 months", "13-24 months", "25-48 months", "49-60 months", "61+ months"]
    engineered["tenure_group"] = pd.cut(
        engineered["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
    ).astype("string")

    engineered["average_monthly_spend"] = np.where(
        engineered["tenure"].gt(0),
        engineered["TotalCharges"] / engineered["tenure"],
        engineered["MonthlyCharges"],
    )
    engineered["average_monthly_spend"] = engineered["average_monthly_spend"].replace(
        [np.inf, -np.inf], np.nan
    )
    engineered["average_monthly_spend"] = engineered["average_monthly_spend"].fillna(
        engineered["MonthlyCharges"]
    )

    engineered["service_count"] = engineered.apply(_count_active_services, axis=1)
    engineered["long_term_customer_flag"] = np.where(engineered["tenure"].ge(24), "Yes", "No")
    high_value_threshold = engineered["MonthlyCharges"].quantile(0.75)
    engineered["high_value_customer_flag"] = np.where(
        engineered["MonthlyCharges"].ge(high_value_threshold), "Yes", "No"
    )
    engineered["month_to_month_flag"] = np.where(
        engineered["Contract"].eq("Month-to-month"), "Yes", "No"
    )
    engineered["has_security_or_support"] = np.where(
        engineered[["OnlineSecurity", "TechSupport"]].eq("Yes").any(axis=1), "Yes", "No"
    )
    engineered["auto_payment_flag"] = np.where(
        engineered["PaymentMethod"].str.contains("automatic", case=False, na=False), "Yes", "No"
    )
    engineered["estimated_clv"] = engineered["average_monthly_spend"] * engineered["tenure"]

    return engineered.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, output_path: Path | str | None = None) -> Path:
    ensure_project_dirs()
    path = Path(output_path) if output_path is not None else PROCESSED_DATA_PATH
    df.to_csv(path, index=False)
    return path


def load_or_prepare_processed_data(raw_csv: Path | str | None = None, force: bool = False) -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists() and not force and raw_csv is None:
        return pd.read_csv(PROCESSED_DATA_PATH)

    raw_df = load_telco_data(raw_csv)
    processed = engineer_features(clean_telco_data(raw_df))
    save_processed_data(processed)
    return processed


def infer_feature_types(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    excluded_columns: Iterable[str] = (ID_COLUMN,),
) -> tuple[list[str], list[str], list[str]]:
    excluded = set(excluded_columns) | {target_column}
    feature_columns = [column for column in df.columns if column not in excluded]
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    return feature_columns, numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    feature_columns, numeric_features, categorical_features = infer_feature_types(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])
    X = df[feature_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
