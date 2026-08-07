"""
feature_extraction.py
Extracts stylometric + sentiment features from cleaned reviews stored in
PostgreSQL (raw_reviews) and pushes them into engineered_features.

Features extracted per review:
- punctuation_freq       : punctuation marks per 100 characters
- vocab_diversity        : type-token ratio (unique words / total words)
- readability_score      : Flesch Reading Ease score (via textstat)
- avg_sentence_length    : average words per sentence
- sentiment_score        : VADER compound sentiment score (-1 to 1)
- rating_sentiment_gap   : mismatch between star rating and text sentiment
"""

import re
import string
import pandas as pd
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine, text

# ---- Update with your own credentials ----
DB_CONFIG = {
    "user": "postgres",
    "password": "Dhrumil2006",
    "host": "localhost",
    "port": "5432",
    "dbname": "reviewshield",
}
CONN_STRING = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

analyzer = SentimentIntensityAnalyzer()


def punctuation_frequency(text_: str) -> float:
    """Punctuation marks per 100 characters."""
    if len(text_) == 0:
        return 0.0
    punct_count = sum(1 for ch in text_ if ch in string.punctuation)
    return round((punct_count / len(text_)) * 100, 4)


def vocab_diversity(text_: str) -> float:
    """Type-token ratio: unique words / total words."""
    words = re.findall(r"\b\w+\b", text_.lower())
    if len(words) == 0:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def avg_sentence_length(text_: str) -> float:
    """Average number of words per sentence."""
    sentences = re.split(r'[.!?]+', text_)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 0:
        return 0.0
    word_counts = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    return round(sum(word_counts) / len(sentences), 4)


def normalize_rating_to_sentiment_scale(rating: float) -> float:
    """
    Maps a 1-5 star rating onto the same -1 to 1 scale as VADER's
    compound sentiment score, so they can be directly compared.
    """
    return round(((rating - 1) / 4) * 2 - 1, 4)


def extract_features(row) -> dict:
    text_ = str(row["review_text"])
    sentiment = analyzer.polarity_scores(text_)["compound"]
    expected_sentiment = normalize_rating_to_sentiment_scale(row["rating"])
    gap = round(abs(sentiment - expected_sentiment), 4)

    return {
        "review_id": row["review_id"],
        "punctuation_freq": punctuation_frequency(text_),
        "vocab_diversity": vocab_diversity(text_),
        "readability_score": round(textstat.flesch_reading_ease(text_), 4),
        "avg_sentence_length": avg_sentence_length(text_),
        "sentiment_score": round(sentiment, 4),
        "rating_sentiment_gap": gap,
    }


def main():
    engine = create_engine(CONN_STRING)

    # Pull cleaned reviews from the DB
    df = pd.read_sql("SELECT review_id, rating, review_text FROM raw_reviews", engine)
    print(f"Loaded {len(df)} reviews from raw_reviews")

    # Extract features row by row
    features = [extract_features(row) for _, row in df.iterrows()]
    features_df = pd.DataFrame(features)
    print(f"Extracted features for {len(features_df)} reviews")
    print(features_df.describe())

    # Push into engineered_features table
    features_df.to_sql("engineered_features", engine, if_exists="append", index=False)

    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO system_logs (stage, message) VALUES (:stage, :msg)"),
            {"stage": "feature_extraction", "msg": f"Extracted and stored features for {len(features_df)} reviews"},
        )

    print(f"Pushed {len(features_df)} rows into engineered_features table")


if __name__ == "__main__":
    main()