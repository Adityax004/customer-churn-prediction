from __future__ import annotations

import sqlite3
from pathlib import Path

from churn_data import load_or_prepare_processed_data
from paths import DB_PATH, SQL_DIR, ensure_project_dirs


def build_database(raw_csv: Path | str | None = None) -> Path:
    """Create a SQLite database from the processed Telco churn dataset."""
    ensure_project_dirs()
    df = load_or_prepare_processed_data(raw_csv=raw_csv, force=raw_csv is not None)

    schema_path = SQL_DIR / "schema.sql"
    with sqlite3.connect(DB_PATH) as connection:
        if schema_path.exists():
            connection.executescript(schema_path.read_text(encoding="utf-8"))
        df.to_sql("customers", connection, if_exists="append", index=False)

    return DB_PATH


if __name__ == "__main__":
    path = build_database()
    print(f"SQLite database created at {path}")
