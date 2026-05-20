-- Customer Churn Prediction and Retention Analysis
-- Run after `python src/run_pipeline.py` creates data/telco_churn.sqlite.

-- 1. Executive churn KPIs
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_recurring_revenue,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM customers;

-- 2. Churn by contract type
SELECT
    Contract,
    COUNT(*) AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY Contract
ORDER BY churn_rate DESC;

-- 3. Churn by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS revenue_at_risk
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate DESC;

-- 4. Tenure cohort risk
SELECT
    tenure_group,
    COUNT(*) AS customers,
    ROUND(AVG(tenure), 1) AS avg_tenure,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- 5. Internet service and support add-on analysis
SELECT
    InternetService,
    OnlineSecurity,
    TechSupport,
    COUNT(*) AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY InternetService, OnlineSecurity, TechSupport
HAVING customers >= 25
ORDER BY churn_rate DESC;

-- 6. High-value customer churn risk
SELECT
    high_value_customer_flag,
    Contract,
    COUNT(*) AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue
FROM customers
GROUP BY high_value_customer_flag, Contract
ORDER BY high_value_customer_flag DESC, churn_rate DESC;

-- 7. Senior citizen churn profile
SELECT
    SeniorCitizen,
    Contract,
    InternetService,
    COUNT(*) AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY SeniorCitizen, Contract, InternetService
HAVING customers >= 20
ORDER BY churn_rate DESC;

-- 8. Service count and churn
SELECT
    service_count,
    COUNT(*) AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY service_count
ORDER BY service_count;
