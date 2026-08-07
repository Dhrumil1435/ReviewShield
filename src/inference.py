"""
inference.py
Production Inference Engine for ReviewShield.
Loads the trained ML pipeline and evaluates arbitrary review text + star ratings in real-time.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
from typing import Dict, Any
from src.config import MODELS_DIR
from src.feature_extraction import (
    punctuation_frequency,
    vocab_diversity,
    avg_sentence_length,
    normalize_rating_to_sentiment_scale,
    analyzer,
)
import textstat

MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"


class ReviewAnalyzer:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.pipeline_data = None
        self.pipeline = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please run src/train_model.py first."
            )
        self.pipeline_data = joblib.load(self.model_path)
        self.pipeline = self.pipeline_data["pipeline"]
        self.feature_names = self.pipeline_data["feature_names"]
        print(f"Loaded {self.pipeline_data['model_name']} model successfully.")

    def extract_single_features(self, review_text: str, rating: float) -> Dict[str, float]:
        text_ = str(review_text)
        sentiment = analyzer.polarity_scores(text_)["compound"]
        expected_sentiment = normalize_rating_to_sentiment_scale(float(rating))
        gap = round(abs(sentiment - expected_sentiment), 4)

        return {
            "punctuation_freq": punctuation_frequency(text_),
            "vocab_diversity": vocab_diversity(text_),
            "readability_score": round(float(textstat.flesch_reading_ease(text_)), 4),
            "avg_sentence_length": avg_sentence_length(text_),
            "sentiment_score": round(sentiment, 4),
            "rating_sentiment_gap": gap,
        }

    def analyze(self, review_text: str, rating: float) -> Dict[str, Any]:
        features = self.extract_single_features(review_text, rating)
        features_df = pd.DataFrame([features])[self.feature_names]

        # Predict probability of class 1 (Deceptive / CG)
        deceptive_prob = float(self.pipeline.predict_proba(features_df)[0, 1])
        is_deceptive = deceptive_prob >= 0.50

        # Identify key risk factors
        risk_factors = []
        if features["rating_sentiment_gap"] > 1.0:
            risk_factors.append(f"High Sentiment-Rating Mismatch (Gap: {features['rating_sentiment_gap']})")
        if features["punctuation_freq"] > 6.0:
            risk_factors.append(f"Elevated Punctuation Density ({features['punctuation_freq']} per 100 chars)")
        if features["vocab_diversity"] < 0.40:
            risk_factors.append(f"Low Vocabulary Diversity ({features['vocab_diversity']} TTR)")
        if features["avg_sentence_length"] > 35.0:
            risk_factors.append(f"Unusually Long Sentences ({features['avg_sentence_length']} words/sent)")

        return {
            "review_text": review_text,
            "rating": rating,
            "deceptive_probability": round(deceptive_prob, 4),
            "deceptive_percentage": round(deceptive_prob * 100, 2),
            "is_deceptive": is_deceptive,
            "classification": "Computer-Generated / Deceptive" if is_deceptive else "Original / Authentic",
            "features": features,
            "risk_factors": risk_factors,
        }


if __name__ == "__main__":
    analyzer_engine = ReviewAnalyzer()
    sample_review = "AMAZING PRODUCT!!!!!! Best item ever bought! Super fast shipping!!!"
    sample_rating = 5.0
    res = analyzer_engine.analyze(sample_review, sample_rating)
    print("\nSample Analysis Result:")
    print(res)
