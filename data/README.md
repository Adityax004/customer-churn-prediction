# Data Directory

This project uses the Kaggle dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The raw Kaggle CSV is not committed to the repository. Place it here:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Expected raw columns:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

Generated files:

- `data/cleaned/telco_churn_processed.csv`
- `data/telco_churn.sqlite`

Data handling notes:

- Blank `TotalCharges` values are converted to numeric and repaired when
  `tenure` indicates a zero-tenure customer.
- The target label `Churn` is encoded during modeling.
- Engineered features include tenure groups, service count, average monthly
  spend, long-term customer flag, and high-value customer flag.
