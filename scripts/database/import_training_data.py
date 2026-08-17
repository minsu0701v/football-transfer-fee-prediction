from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


# ============================================================
# Path
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "training_dataset_european.csv"
)


# ============================================================
# PostgreSQL
# ============================================================

DB_USER = "postgres"
DB_PASSWORD = "root"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "football_transfer_db"


DATABASE_URL = (
    f"postgresql+psycopg://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


engine = create_engine(
    DATABASE_URL
)


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

print("=" * 60)
print("CSV")
print("=" * 60)

print(
    f"Path    : {CSV_PATH}"
)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)


# ============================================================
# Import PostgreSQL
# ============================================================

df.to_sql(
    name="training_data",
    con=engine,
    if_exists="replace",
    index=False,
    chunksize=1000,
)


# ============================================================
# Validation
# ============================================================

with engine.connect() as connection:

    result = connection.execute(
        text(
            "SELECT COUNT(*) "
            "FROM training_data"
        )
    )

    db_count = result.scalar()


print()
print("=" * 60)
print("PostgreSQL")
print("=" * 60)

print(
    f"Rows    : {db_count:,}"
)

if len(df) == db_count:
    print("CSV / PostgreSQL row count 일치")
else:
    print("WARNING: row count 불일치")