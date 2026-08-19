"""
db_setup.py
Initializes the ReviewShield PostgreSQL database tables and seeds raw reviews idempotently.
Supports schema migrations for RoBERTa + VADER features.
"""

import pandas as pd
from sqlalchemy import text
from src.config import get_db_engine, CLEANED_DATA_PATH, RAW_DATA_PATH

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS raw_reviews (
    review_id SERIAL PRIMARY KEY,
    category VARCHAR(100),
    rating NUMERIC(2, 1),
    label VARCHAR(10),
    review_text TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS engineered_features (
    feature_id SERIAL PRIMARY KEY,
    review_id INT REFERENCES raw_reviews(review_id) ON DELETE CASCADE,
    punctuation_freq FLOAT,
    vocab_diversity FLOAT,
    readability_score FLOAT,
    avg_sentence_length FLOAT,
    sentiment_score FLOAT,
    rating_sentiment_gap FLOAT,
    roberta_sentiment FLOAT,
    roberta_rating_sentiment_gap FLOAT,
    vader_roberta_dissonance FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_review_feature UNIQUE (review_id)
);

CREATE TABLE IF NOT EXISTS system_logs (
    log_id SERIAL PRIMARY KEY,
    stage VARCHAR(100),
    message TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

MIGRATION_SQL = """
ALTER TABLE engineered_features ADD COLUMN IF NOT EXISTS roberta_sentiment FLOAT;
ALTER TABLE engineered_features ADD COLUMN IF NOT EXISTS roberta_rating_sentiment_gap FLOAT;
ALTER TABLE engineered_features ADD COLUMN IF NOT EXISTS vader_roberta_dissonance FLOAT;
"""


def init_db():
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLES_SQL))
        conn.execute(text(MIGRATION_SQL))
        conn.execute(
            text("INSERT INTO system_logs (stage, message) VALUES (:stage, :msg)"),
            {"stage": "db_setup", "msg": "Database schemas & RoBERTa column migration verified successfully."},
        )
    print("Database schemas & RoBERTa column migrations updated successfully.")


def seed_raw_reviews():
    engine = get_db_engine()

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM raw_reviews")).scalar()
        if count > 0:
            print(f"raw_reviews table already populated ({count} records). Skipping seed.")
            return

    data_path = CLEANED_DATA_PATH if CLEANED_DATA_PATH.exists() else RAW_DATA_PATH
    if not data_path.exists():
        print(f"Data file not found at {data_path}. Skipping seed.")
        return

    print(f"Seeding raw_reviews from {data_path}...")
    df = pd.read_csv(data_path)

    column_mapping = {
        "category": "category",
        "rating": "rating",
        "label": "label",
        "text_": "review_text",
        "review_text": "review_text",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    expected_cols = ["category", "rating", "label", "review_text"]
    df = df[[col for col in expected_cols if col in df.columns]]

    df.to_sql("raw_reviews", engine, if_exists="append", index=False)

    with engine.begin() as conn:
        seeded_count = conn.execute(text("SELECT COUNT(*) FROM raw_reviews")).scalar()
        conn.execute(
            text("INSERT INTO system_logs (stage, message) VALUES (:stage, :msg)"),
            {"stage": "db_setup", "msg": f"Seeded {seeded_count} rows into raw_reviews."},
        )

    print(f"Successfully seeded {seeded_count} reviews into raw_reviews.")


if __name__ == "__main__":
    init_db()
    seed_raw_reviews()
