DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customerID TEXT PRIMARY KEY,
    gender TEXT,
    SeniorCitizen TEXT,
    Partner TEXT,
    Dependents TEXT,
    tenure REAL,
    PhoneService TEXT,
    MultipleLines TEXT,
    InternetService TEXT,
    OnlineSecurity TEXT,
    OnlineBackup TEXT,
    DeviceProtection TEXT,
    TechSupport TEXT,
    StreamingTV TEXT,
    StreamingMovies TEXT,
    Contract TEXT,
    PaperlessBilling TEXT,
    PaymentMethod TEXT,
    MonthlyCharges REAL,
    TotalCharges REAL,
    Churn TEXT,
    tenure_group TEXT,
    average_monthly_spend REAL,
    service_count INTEGER,
    long_term_customer_flag TEXT,
    high_value_customer_flag TEXT,
    month_to_month_flag TEXT,
    has_security_or_support TEXT,
    auto_payment_flag TEXT,
    estimated_clv REAL
);

CREATE INDEX idx_customers_churn ON customers (Churn);
CREATE INDEX idx_customers_contract ON customers (Contract);
CREATE INDEX idx_customers_payment_method ON customers (PaymentMethod);
CREATE INDEX idx_customers_internet_service ON customers (InternetService);
CREATE INDEX idx_customers_tenure_group ON customers (tenure_group);
CREATE INDEX idx_customers_high_value ON customers (high_value_customer_flag);
