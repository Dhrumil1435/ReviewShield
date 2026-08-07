"""
feature_extraction.py
Extracts stylometric + sentiment features from cleaned reviews stored in
PostgreSQL (raw_reviews) and pushes them into engineered_features idempotently.

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
from sqlalchemy import text
from src.config import get_db_engine

analyzer = SentimentIntensityAnalyzer()


def punctuation_frequency(text_: str) -> float:
    """Punctuation marks per 100 characters."""
    if not text_:
        return 0.0
    punct_count = sum(1 for ch in text_ if ch in string.punctuation)
    return round((punct_count / len(text_)) * 100, 4)


def vocab_diversity(text_: str) -> float:
    """Type-token ratio: unique words / total words."""
    words = re.findall(r"\b\w+\b", text_.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def avg_sentence_length(text_: str) -> float:
    """Average number of words per sentence."""
    sentences = re.split(r"[.!?]+", text_)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    word_counts = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    return round(sum(word_counts) / len(sentences), 4)


def normalize_rating_to_sentiment_scale(rating: float) -> float:
    """
    Maps a 1-5 star rating onto the same -1 to 1 scale as VADER's
    compound sentiment score.
    """
    return round(((rating - 1) / 4) * 2 - 1, 4)


def extract_features(row) -> dict:
    text_ = str(row["review_text"])
    sentiment = analyzer.polarity_scores(text_)["compound"]
    expected_sentiment = normalize_rating_to_sentiment_scale(float(row["rating"]))
    gap = round(abs(sentiment - expected_sentiment), 4)

    return {
        "review_id": int(row["review_id"]),
        "punctuation_freq": punctuation_frequency(text_),
        "vocab_diversity": vocab_diversity(text_),
        "readability_score": round(float(textstat.flesch_reading_ease(text_)), 4),
        "avg_sentence_length": avg_sentence_length(text_),
        "sentiment_score": round(sentiment, 4),
        "rating_sentiment_gap": gap,
    }


def run_feature_extraction(batch_size: int = 5000):
    engine = get_db_engine()

    # Find unextracted review IDs (Idempotent execution)
    query_unprocessed = """
        SELECT r.review_id, r.rating, r.review_text 
        FROM raw_reviews r
        LEFT JOIN engineered_features f ON r.review_id = f.review_id
        WHERE f.review_id IS NULL
    """
    df = pd.read_sql(query_unprocessed, engine)
    total_unprocessed = len(df)

    if total_unprocessed == 0:
        print("All reviews in raw_reviews already have extracted features in engineered_features.")
        return

    print(f"Extracting features for {total_unprocessed} remaining reviews...")

    # Process in batches
    for i in range(0, total_unprocessed, batch_size):
        batch_df = df.iloc[i : i + batch_size]
        features = [extract_features(row) for _, row in batch_df.iterrows()]
        features_df = pd.DataFrame(features)

        features_df.to_sql("engineered_features", engine, if_exists="append", index=False)
        print(f"Pushed batch {i // batch_size + 1} ({len(features_df)} rows) to engineered_features.")

    # Log completion in system_logs
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO system_logs (stage, message) VALUES (:stage, :msg)"),
            {
                "stage": "feature_extraction",
                "msg": f"Successfully extracted features for {total_unprocessed} reviews.",
            },
        )

    print(f"Feature extraction pipeline complete. Extracted {total_unprocessed} reviews.")


if __name__ == "__main__":
    run_feature_extraction()