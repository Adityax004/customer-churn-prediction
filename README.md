# Customer Churn Prediction and Retention Analysis

End-to-end data analytics and machine learning project for the Kaggle
Telco Customer Churn dataset. The project simulates a real telecom
retention problem: identify customers likely to discontinue service,
explain the drivers of churn, and translate model results into targeted
retention strategies.

Dataset source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Business Problem

Telecom providers operate in a highly competitive market where acquiring a
new customer is often more expensive than retaining an existing one. Churn
creates revenue leakage, increases acquisition pressure, reduces customer
lifetime value, and can signal weaknesses in pricing, service quality, or
customer experience.

The objective is to predict whether a customer is likely to churn and
identify the factors most associated with churn risk.

Business goals:

- Detect high-risk customers before they cancel service.
- Prioritize retention campaigns toward customers with the highest expected
  value and churn risk.
- Understand the impact of contract type, tenure, monthly charges, services,
  payment method, and support-related add-ons.
- Recommend practical retention actions by customer segment.

Key KPIs:

- Overall churn rate
- Recall for churn customers
- ROC-AUC
- Retention campaign target size
- High-risk customer count
- Estimated monthly recurring revenue at risk
- Churn rate by contract, tenure group, payment method, and service bundle

Expected outcomes:

- Clean, reproducible analytics workflow.
- EDA with business interpretation after each major visualization.
- Classification models tuned with cross-validation.
- Model comparison focused on recall and ROC-AUC.
- Explainability outputs for feature importance and churn drivers.
- Segment-wise retention recommendations.

## Repository Structure

```text
.
|-- data/
|   |-- raw/
|   |-- cleaned/
|   `-- README.md
|-- dashboard/
|   |-- powerbi_dashboard_spec.md
|   |-- powerbi_measures.dax
|   `-- streamlit_app.py
|-- models/
|-- notebooks/
|   `-- customer_churn_prediction_and_retention_analysis.ipynb
|-- reports/
|   `-- business_insights_summary.md
|-- sql/
|   |-- schema.sql
|   `-- analysis_queries.sql
|-- src/
|   |-- churn_data.py
|   |-- churn_modeling.py
|   |-- churn_visuals.py
|   |-- download_kaggle_data.py
|   |-- build_sqlite.py
|   |-- paths.py
|   `-- run_pipeline.py
|-- visuals/
|-- README.md
`-- requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

This repo does not redistribute Kaggle data. Download the dataset and place
the CSV here:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

With Kaggle API credentials configured, you can also run:

```powershell
python src/download_kaggle_data.py
```

## Run the Project

Run the full preprocessing, EDA, modeling, reporting, and SQLite export
pipeline:

```powershell
python src/run_pipeline.py
```

Useful options:

```powershell
python src/run_pipeline.py --skip-models
python src/run_pipeline.py --skip-visuals
python src/run_pipeline.py --raw-csv data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Generated outputs:

- `data/cleaned/telco_churn_processed.csv`
- `reports/model_metrics.csv`
- `reports/test_set_predictions.csv`
- `reports/feature_importance.csv`
- `reports/eda_business_insights.md`
- `models/best_churn_model.joblib`
- `visuals/*.png`
- `data/telco_churn.sqlite`

## Notebook Workflow

Open the main notebook:

```text
notebooks/customer_churn_prediction_and_retention_analysis.ipynb
```

The notebook covers:

- Business objective, KPIs, and impact of churn
- Dataset overview and data quality checks
- Missing value handling and datatype fixes
- Label encoding, one-hot encoding, scaling, and class imbalance handling
- EDA with churn interpretations
- Feature engineering
- Logistic Regression, Decision Tree, Random Forest, and optional XGBoost
- Hyperparameter tuning with `GridSearchCV`
- Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and ROC curves
- Feature importance, Logistic Regression coefficients, and optional SHAP workflow
- Retention strategy recommendations

## Dashboard

Streamlit dashboard:

```powershell
streamlit run dashboard/streamlit_app.py
```

Power BI implementation guidance is documented in
`dashboard/powerbi_dashboard_spec.md`, with starter DAX measures in
`dashboard/powerbi_measures.dax`.

Dashboard pages include:

- Churn overview KPIs
- Churn risk segmentation
- Customer demographics
- Contract, billing, and service analysis
- Feature importance and model prediction insights

## Modeling Approach

The pipeline uses a business-first classification workflow:

- `TotalCharges` is converted to numeric and blank records are repaired where
  tenure indicates a zero-charge customer.
- Categorical features are one-hot encoded.
- The churn label is label encoded.
- Numerical variables are standardized.
- Class imbalance is addressed with class weighting by default.
- Models are tuned with stratified cross-validation.
- The best model is selected by recall first and ROC-AUC second.

Recall is emphasized because missing a likely churner can mean losing a
customer entirely. False positives still have a cost, but they can often be
managed through lower-cost offers, loyalty messages, or service outreach.

## Resume-Ready Summary

Built an end-to-end telecom churn analytics project using Python, Pandas,
Seaborn, Scikit-learn, optional XGBoost/SHAP, SQL, and Streamlit/Power BI.
Cleaned and engineered Telco customer data, analyzed churn drivers, trained
and tuned classification models, evaluated recall and ROC-AUC, explained
feature importance, and designed actionable retention strategies for
high-risk customer segments.
