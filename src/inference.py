"""
inference.py
Production Inference Engine for ReviewShield.
Loads the trained ML pipeline (Stylometric + VADER + RoBERTa) and evaluates arbitrary review text + star ratings in real-time or batch CSV data.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
from typing import Dict, Any, List
from src.config import MODELS_DIR
from src.feature_extraction import (
    punctuation_frequency,
    vocab_diversity,
    avg_sentence_length,
    normalize_rating_to_sentiment_scale,
    vader_analyzer,
    roberta_engine,
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
        
        # VADER Sentiment
        vader_sent = vader_analyzer.polarity_scores(text_)["compound"]
        expected_sentiment = normalize_rating_to_sentiment_scale(float(rating))
        vader_gap = round(abs(vader_sent - expected_sentiment), 4)

        # RoBERTa Contextual Sentiment
        roberta_res = roberta_engine.analyze_text(text_)
        roberta_sent = roberta_res["roberta_compound"]
        roberta_gap = round(abs(roberta_sent - expected_sentiment), 4)

        # VADER vs RoBERTa Dissonance
        dissonance = round(abs(vader_sent - roberta_sent), 4)

        return {
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

    def analyze(self, review_text: str, rating: float) -> Dict[str, Any]:
        features = self.extract_single_features(review_text, rating)
        features_df = pd.DataFrame([features])[self.feature_names]

        deceptive_prob = float(self.pipeline.predict_proba(features_df)[0, 1])
        is_deceptive = deceptive_prob >= 0.50

        risk_factors = []
        if features["vader_roberta_dissonance"] > 0.65:
            risk_factors.append(f"High VADER-RoBERTa Dissonance ({features['vader_roberta_dissonance']:.2f})")
        if features["roberta_rating_sentiment_gap"] > 1.0:
            risk_factors.append(f"High RoBERTa Sentiment Mismatch (Gap: {features['roberta_rating_sentiment_gap']:.2f})")
        if features["rating_sentiment_gap"] > 1.0:
            risk_factors.append(f"High VADER Sentiment Mismatch (Gap: {features['rating_sentiment_gap']:.2f})")
        if features["punctuation_freq"] > 6.0:
            risk_factors.append(f"Elevated Punctuation Density ({features['punctuation_freq']:.2f}/100 chars)")
        if features["vocab_diversity"] < 0.40:
            risk_factors.append(f"Low Vocabulary Diversity ({features['vocab_diversity']:.2f} TTR)")

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

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str, rating_col: str, progress_callback=None) -> pd.DataFrame:
        """
        Batch processes a pandas DataFrame containing review text and star ratings.
        Returns the original DataFrame enriched with deceptive risk metrics and classifications.
        """
        results = []
        total_rows = len(df)

        for idx, (_, row) in enumerate(df.iterrows()):
            review_text = row[text_col]
            rating = row[rating_col]
            analysis = self.analyze(review_text, rating)

            results.append({
                "Deceptive Probability (%)": analysis["deceptive_percentage"],
                "Classification": "DECEPTIVE" if analysis["is_deceptive"] else "GENUINE",
                "Risk Factors": ", ".join(analysis["risk_factors"]) if analysis["risk_factors"] else "None",
                "VADER Sentiment": analysis["features"]["sentiment_score"],
                "RoBERTa Sentiment": analysis["features"]["roberta_sentiment"],
                "VADER-RoBERTa Dissonance": analysis["features"]["vader_roberta_dissonance"],
                "Vocab Diversity (TTR)": analysis["features"]["vocab_diversity"],
                "Readability Score": analysis["features"]["readability_score"],
            })

            if progress_callback:
                progress_callback((idx + 1) / total_rows)

        results_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), results_df], axis=1)


if __name__ == "__main__":
    analyzer_engine = ReviewAnalyzer()
    sample_review = "AMAZING PRODUCT!!!!!! Best item ever bought! Super fast shipping!!!"
    sample_rating = 5.0
    res = analyzer_engine.analyze(sample_review, sample_rating)
    print("\nSample Analysis Result:")
    print(res)
