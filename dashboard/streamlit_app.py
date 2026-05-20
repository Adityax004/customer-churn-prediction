from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from churn_data import clean_telco_data, engineer_features, load_telco_data, load_or_prepare_processed_data
from paths import FEATURE_IMPORTANCE_PATH, MODEL_PATH, PREDICTIONS_PATH, PROCESSED_DATA_PATH


st.set_page_config(
    page_title="Customer Churn Prediction and Retention Analysis",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="Set2")


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    raw_df = load_telco_data()
    return engineer_features(clean_telco_data(raw_df))


@st.cache_resource
def load_model_artifact():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def add_heuristic_risk(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    risk = pd.Series(0.0, index=scored.index)
    risk += scored["Contract"].eq("Month-to-month") * 0.25
    risk += scored["tenure"].le(12) * 0.20
    risk += scored["PaymentMethod"].str.contains("Electronic check", case=False, na=False) * 0.15
    risk += scored["InternetService"].eq("Fiber optic") * 0.10
    risk += scored["OnlineSecurity"].ne("Yes") * 0.10
    risk += scored["TechSupport"].ne("Yes") * 0.10
    risk += scored["high_value_customer_flag"].eq("Yes") * 0.10
    scored["churn_probability"] = risk.clip(0, 0.95)
    return scored


def score_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    artifact = load_model_artifact()
    if artifact is None:
        return add_heuristic_risk(df), "Heuristic risk score"

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    scored = df.copy()
    scored["churn_probability"] = model.predict_proba(scored[feature_columns])[:, 1]
    return scored, f"Model: {artifact['model_name']}"


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        contract = st.multiselect("Contract", sorted(df["Contract"].dropna().unique()))
        internet = st.multiselect("Internet Service", sorted(df["InternetService"].dropna().unique()))
        payment = st.multiselect("Payment Method", sorted(df["PaymentMethod"].dropna().unique()))
        senior = st.multiselect("Senior Citizen", sorted(df["SeniorCitizen"].dropna().unique()))

    filtered = df.copy()
    if contract:
        filtered = filtered[filtered["Contract"].isin(contract)]
    if internet:
        filtered = filtered[filtered["InternetService"].isin(internet)]
    if payment:
        filtered = filtered[filtered["PaymentMethod"].isin(payment)]
    if senior:
        filtered = filtered[filtered["SeniorCitizen"].isin(senior)]
    return filtered


def show_metric_row(df: pd.DataFrame) -> None:
    total_customers = len(df)
    churn_rate = df["Churn"].eq("Yes").mean() if total_customers else 0
    mrr = df["MonthlyCharges"].sum()
    revenue_at_risk = df.loc[df["Churn"].eq("Yes"), "MonthlyCharges"].sum()
    high_risk_customers = int(df["churn_probability"].ge(0.60).sum())

    cols = st.columns(5)
    cols[0].metric("Customers", f"{total_customers:,.0f}")
    cols[1].metric("Churn Rate", f"{churn_rate:.1%}")
    cols[2].metric("Monthly Revenue", f"${mrr:,.0f}")
    cols[3].metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")
    cols[4].metric("High Risk", f"{high_risk_customers:,.0f}")


def plot_churn_by_category(df: pd.DataFrame, column: str, title: str) -> None:
    chart_df = (
        df.assign(churn_flag=df["Churn"].eq("Yes").astype(int))
        .groupby(column)
        .agg(customers=("churn_flag", "size"), churn_rate=("churn_flag", "mean"))
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=chart_df, x=column, y="churn_rate", ax=ax, color="#cc4125")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Churn Rate")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)


def show_feature_importance() -> None:
    st.subheader("Model Drivers")
    if not FEATURE_IMPORTANCE_PATH.exists():
        st.info("Run `python src/run_pipeline.py` to generate feature importance.")
        return

    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(15)
    importance["feature"] = (
        importance["feature"]
        .str.replace("categorical__", "", regex=False)
        .str.replace("numeric__", "", regex=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importance, y="feature", x="absolute_importance", ax=ax, color="#2f6f9f")
    ax.set_xlabel("Absolute Importance")
    ax.set_ylabel("")
    st.pyplot(fig)


def show_prediction_table(df: pd.DataFrame) -> None:
    st.subheader("High-Risk Customer Watchlist")
    display_cols = [
        "customerID",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "churn_probability",
    ]
    available_cols = [column for column in display_cols if column in df.columns]
    watchlist = df.sort_values("churn_probability", ascending=False)[available_cols].head(25)
    st.dataframe(
        watchlist.style.format({"MonthlyCharges": "${:,.2f}", "churn_probability": "{:.1%}"}),
        use_container_width=True,
    )


def main() -> None:
    st.title("Customer Churn Prediction and Retention Analysis")

    try:
        df = load_dashboard_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    scored_df, scoring_source = score_customers(df)
    filtered = filter_data(scored_df)

    st.caption(scoring_source)
    show_metric_row(filtered)

    tab_overview, tab_segments, tab_model, tab_actions = st.tabs(
        ["Overview", "Risk Segments", "Model Insights", "Retention Actions"]
    )

    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            plot_churn_by_category(filtered, "Contract", "Churn Rate by Contract")
        with col2:
            plot_churn_by_category(filtered, "tenure_group", "Churn Rate by Tenure Group")

        col3, col4 = st.columns(2)
        with col3:
            plot_churn_by_category(filtered, "PaymentMethod", "Churn Rate by Payment Method")
        with col4:
            plot_churn_by_category(filtered, "InternetService", "Churn Rate by Internet Service")

    with tab_segments:
        risk_bins = pd.cut(
            filtered["churn_probability"],
            bins=[0, 0.30, 0.60, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
        segment = (
            filtered.assign(risk_band=risk_bins)
            .groupby("risk_band", observed=False)
            .agg(
                customers=("customerID", "count"),
                avg_probability=("churn_probability", "mean"),
                monthly_revenue=("MonthlyCharges", "sum"),
            )
            .reset_index()
        )
        st.dataframe(
            segment.style.format({"avg_probability": "{:.1%}", "monthly_revenue": "${:,.0f}"}),
            use_container_width=True,
        )
        show_prediction_table(filtered)

    with tab_model:
        show_feature_importance()
        if PREDICTIONS_PATH.exists():
            predictions = pd.read_csv(PREDICTIONS_PATH)
            st.subheader("Prediction Sample")
            st.dataframe(predictions.head(20), use_container_width=True)

    with tab_actions:
        st.subheader("Recommended Retention Plays")
        actions = pd.DataFrame(
            [
                ["New customers", "Tenure <= 12 months", "Onboarding check-ins, first-bill support, welcome credits"],
                ["High-charge users", "High monthly charges", "Bill review, right-plan offer, premium support callback"],
                ["Month-to-month", "No long-term contract", "Annual contract incentive, loyalty bundle, price-lock offer"],
                ["Senior citizens", "SeniorCitizen = Yes", "Simplified support, proactive service assistance, trust-building outreach"],
                ["No security/support", "No OnlineSecurity or TechSupport", "Bundle security/support trial with service education"],
            ],
            columns=["Segment", "Trigger", "Recommended Action"],
        )
        st.dataframe(actions, use_container_width=True)


if __name__ == "__main__":
    main()
