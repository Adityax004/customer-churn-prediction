from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from churn_data import TARGET_COLUMN, load_or_prepare_processed_data
from paths import EDA_INSIGHTS_PATH, VISUALS_DIR, ensure_project_dirs


sns.set_theme(style="whitegrid", palette="Set2")


def _save_current_plot(filename: str) -> Path:
    ensure_project_dirs()
    path = VISUALS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        df.assign(churn_flag=df[TARGET_COLUMN].eq("Yes").astype(int))
        .groupby(column, dropna=False)
        .agg(customers=("churn_flag", "size"), churn_rate=("churn_flag", "mean"))
        .sort_values("churn_rate", ascending=False)
        .reset_index()
    )
    return summary


def plot_churn_distribution(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, order=["No", "Yes"])
    total = len(df)
    for patch in ax.patches:
        count = patch.get_height()
        ax.annotate(
            f"{count:,.0f}\n{count / total:.1%}",
            (patch.get_x() + patch.get_width() / 2, count),
            ha="center",
            va="bottom",
        )
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Customers")
    return _save_current_plot("churn_distribution.png")


def plot_stacked_churn_rate(df: pd.DataFrame, column: str, filename: str, title: str) -> Path:
    counts = pd.crosstab(df[column], df[TARGET_COLUMN], normalize="index")
    counts = counts.reindex(columns=["No", "Yes"]).fillna(0)
    counts.plot(kind="bar", stacked=True, figsize=(9, 5), color=["#6aa84f", "#cc4125"])
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Share of Customers")
    plt.legend(title="Churn", loc="upper right")
    plt.xticks(rotation=30, ha="right")
    return _save_current_plot(filename)


def plot_monthly_charges_distribution(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="MonthlyCharges",
        hue=TARGET_COLUMN,
        bins=35,
        kde=True,
        stat="density",
        common_norm=False,
    )
    plt.title("Monthly Charges Distribution by Churn")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Density")
    return _save_current_plot("monthly_charges_by_churn.png")


def plot_monthly_charges_boxplot(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="MonthlyCharges", order=["No", "Yes"])
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    return _save_current_plot("monthly_charges_boxplot.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    numeric_df = df.select_dtypes(include="number").copy()
    numeric_df["ChurnFlag"] = df[TARGET_COLUMN].eq("Yes").astype(int)
    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5)
    plt.title("Correlation Matrix for Numeric Features")
    return _save_current_plot("correlation_matrix.png")


def build_eda_insights(df: pd.DataFrame) -> str:
    churn_rate = df[TARGET_COLUMN].eq("Yes").mean()
    lines = [
        "# EDA Business Insights",
        "",
        f"- Overall churn rate: {churn_rate:.1%}.",
        "- Churn prevention should prioritize recall because missed churners represent lost recurring revenue.",
    ]

    for column in [
        "Contract",
        "PaymentMethod",
        "InternetService",
        "tenure_group",
        "SeniorCitizen",
        "high_value_customer_flag",
    ]:
        if column not in df.columns:
            continue
        rates = churn_rate_by(df, column)
        top = rates.iloc[0]
        lines.append(
            f"- Highest churn segment for {column}: {top[column]} "
            f"({top['churn_rate']:.1%}, {int(top['customers']):,} customers)."
        )

    low_tenure_churn = churn_rate_by(df, "tenure_group").head(1)
    if not low_tenure_churn.empty:
        lines.append(
            "- Tenure segmentation is a strong early-warning view because newly acquired customers "
            "often churn before lifetime value has been recovered."
        )

    lines.extend(
        [
            "",
            "Recommended EDA follow-ups:",
            "",
            "- Compare month-to-month customers by payment method and monthly charges.",
            "- Track high-value customers without online security or tech support as a premium-risk segment.",
            "- Monitor onboarding cohorts in the first year of tenure.",
        ]
    )
    return "\n".join(lines)


def generate_visuals(raw_csv: Path | str | None = None) -> list[Path]:
    df = load_or_prepare_processed_data(raw_csv=raw_csv, force=raw_csv is not None)
    outputs = [
        plot_churn_distribution(df),
        plot_stacked_churn_rate(df, "Contract", "churn_by_contract.png", "Churn Share by Contract Type"),
        plot_stacked_churn_rate(
            df,
            "PaymentMethod",
            "churn_by_payment_method.png",
            "Churn Share by Payment Method",
        ),
        plot_stacked_churn_rate(
            df,
            "InternetService",
            "churn_by_internet_service.png",
            "Churn Share by Internet Service",
        ),
        plot_stacked_churn_rate(
            df,
            "tenure_group",
            "churn_by_tenure_group.png",
            "Churn Share by Tenure Group",
        ),
        plot_stacked_churn_rate(
            df,
            "SeniorCitizen",
            "churn_by_senior_citizen.png",
            "Churn Share by Senior Citizen Status",
        ),
        plot_monthly_charges_distribution(df),
        plot_monthly_charges_boxplot(df),
        plot_correlation_heatmap(df),
    ]

    EDA_INSIGHTS_PATH.write_text(build_eda_insights(df), encoding="utf-8")
    return outputs


if __name__ == "__main__":
    generated = generate_visuals()
    print("Generated visuals:")
    for output in generated:
        print(f"- {output}")
