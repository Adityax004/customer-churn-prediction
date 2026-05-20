from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "cleaned"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALS_DIR = PROJECT_ROOT / "visuals"
MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

RAW_DATA_PATH = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = CLEAN_DIR / "telco_churn_processed.csv"
DB_PATH = DATA_DIR / "telco_churn.sqlite"
MODEL_PATH = MODELS_DIR / "best_churn_model.joblib"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
PREDICTIONS_PATH = REPORTS_DIR / "test_set_predictions.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
EDA_INSIGHTS_PATH = REPORTS_DIR / "eda_business_insights.md"


def ensure_project_dirs() -> None:
    for path in (RAW_DIR, CLEAN_DIR, REPORTS_DIR, VISUALS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
