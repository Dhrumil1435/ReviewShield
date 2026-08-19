"""
feature_extraction.py
Extracts stylometric + VADER + RoBERTa sentiment features from cleaned reviews stored in
PostgreSQL (raw_reviews) and pushes them into engineered_features idempotently.

Features extracted per review:
- punctuation_freq          : punctuation marks per 100 characters
- vocab_diversity           : type-token ratio (unique words / total words)
- readability_score         : Flesch Reading Ease score (via textstat)
- avg_sentence_length       : average words per sentence
- sentiment_score           : VADER compound sentiment score (-1 to 1)
- rating_sentiment_gap      : VADER rating-sentiment gap
- roberta_sentiment         : RoBERTa contextual sentiment score (-1 to 1)
- roberta_rating_sentiment_gap : RoBERTa rating-sentiment gap
- vader_roberta_dissonance  : absolute difference between VADER and RoBERTa sentiment
"""

import re
import sys
import string
import pandas as pd
import textstat
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import text
from src.config import get_db_engine
from src.roberta_engine import RobertaSentimentEngine

vader_analyzer = SentimentIntensityAnalyzer()
roberta_engine = RobertaSentimentEngine()


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
    Maps a 1-5 star rating onto the same -1 to 1 scale as VADER/RoBERTa
    compound sentiment score.
    """
    return round(((rating - 1) / 4) * 2 - 1, 4)


def extract_features(row) -> dict:
    text_ = str(row["review_text"])

    # VADER Sentiment
    vader_sent = vader_analyzer.polarity_scores(text_)["compound"]
    expected_sentiment = normalize_rating_to_sentiment_scale(float(row["rating"]))
    vader_gap = round(abs(vader_sent - expected_sentiment), 4)

    # RoBERTa Deep Learning Contextual Sentiment
    roberta_res = roberta_engine.analyze_text(text_)
    roberta_sent = roberta_res["roberta_compound"]
    roberta_gap = round(abs(roberta_sent - expected_sentiment), 4)

    # VADER vs RoBERTa Dissonance Score
    dissonance = round(abs(vader_sent - roberta_sent), 4)

    return {
        "review_id": int(row["review_id"]),
        "punctuation_freq": punctuation_frequency(text_),
        "vocab_diversity": vocab_diversity(text_),
        "readability_score": round(float(textstat.flesch_reading_ease(text_)), 4),
        "avg_sentence_length": avg_sentence_length(text_),
        "sentiment_score": round(vader_sent, 4),
        "rating_sentiment_gap": vader_gap,
        "roberta_sentiment": round(roberta_sent, 4),
        "roberta_rating_sentiment_gap": roberta_gap,
        "vader_roberta_dissonance": dissonance,
    }


def run_feature_extraction(batch_size: int = 2000):
    engine = get_db_engine()

    query_unprocessed = """
        SELECT r.review_id, r.rating, r.review_text 
        FROM raw_reviews r
        LEFT JOIN engineered_features f ON r.review_id = f.review_id
        WHERE f.review_id IS NULL OR f.roberta_sentiment IS NULL
    """
    df = pd.read_sql(query_unprocessed, engine)
    total_unprocessed = len(df)

    if total_unprocessed == 0:
        print("All reviews in raw_reviews already have extracted VADER + RoBERTa features.")
        return

    print(f"Extracting VADER + RoBERTa features for {total_unprocessed} remaining reviews...")

    for i in range(0, total_unprocessed, batch_size):
        batch_df = df.iloc[i : i + batch_size]
        features = [extract_features(row) for _, row in batch_df.iterrows()]

        with engine.begin() as conn:
            for feat in features:
                conn.execute(
                    text("""
                        INSERT INTO engineered_features 
                        (review_id, punctuation_freq, vocab_diversity, readability_score, 
                         avg_sentence_length, sentiment_score, rating_sentiment_gap, 
                         roberta_sentiment, roberta_rating_sentiment_gap, vader_roberta_dissonance)
                        VALUES (:review_id, :punctuation_freq, :vocab_diversity, :readability_score,
                                :avg_sentence_length, :sentiment_score, :rating_sentiment_gap,
                                :roberta_sentiment, :roberta_rating_sentiment_gap, :vader_roberta_dissonance)
                        ON CONFLICT (review_id) DO UPDATE SET
                            roberta_sentiment = EXCLUDED.roberta_sentiment,
                            roberta_rating_sentiment_gap = EXCLUDED.roberta_rating_sentiment_gap,
                            vader_roberta_dissonance = EXCLUDED.vader_roberta_dissonance;
                    """),
                    feat,
                )
        print(f"Pushed batch {i // batch_size + 1} ({len(batch_df)} rows) into engineered_features.")

    print(f"RoBERTa + VADER feature extraction pipeline complete.")


if __name__ == "__main__":
    run_feature_extraction()