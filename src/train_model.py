"""
train_model.py
Trains, evaluates, and serializes machine learning models for ReviewShield deceptive review detection.
Hybrid Architecture: Combines 9 Stylometric/VADER/RoBERTa features with TF-IDF Word N-Grams.
Models evaluated: Logistic Regression, Random Forest, SGD Classifier.
Saves the best performing hybrid pipeline to models/best_model_pipeline.joblib.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from src.config import get_db_engine, MODELS_DIR

NUMERICAL_COLS = [
    "punctuation_freq",
    "vocab_diversity",
    "readability_score",
    "avg_sentence_length",
    "sentiment_score",
    "rating_sentiment_gap",
    "roberta_sentiment",
    "roberta_rating_sentiment_gap",
    "vader_roberta_dissonance",
]


def load_dataset():
    engine = get_db_engine()
    query = """
        SELECT f.review_id, r.review_text, r.label, f.punctuation_freq, f.vocab_diversity, 
               f.readability_score, f.avg_sentence_length, f.sentiment_score, f.rating_sentiment_gap,
               COALESCE(f.roberta_sentiment, 0.0) as roberta_sentiment,
               COALESCE(f.roberta_rating_sentiment_gap, 0.0) as roberta_rating_sentiment_gap,
               COALESCE(f.vader_roberta_dissonance, 0.0) as vader_roberta_dissonance
        FROM engineered_features f
        JOIN raw_reviews r ON f.review_id = r.review_id
    """
    df = pd.read_sql(query, engine)
    
    # Ensure string type for text
    df["review_text"] = df["review_text"].fillna("").astype(str)
    
    # Encode Target: CG (Computer Generated / Deceptive) = 1, OR (Original / Genuine) = 0
    df["target"] = df["label"].apply(lambda x: 1 if str(x).strip().upper() in ["CG", "1", "FAKE"] else 0)
    
    print(f"Dataset loaded successfully: {len(df)} total reviews.")
    print(f"Target distribution (1=Deceptive, 0=Genuine):\n{df['target'].value_counts()}")
    return df


def train_and_evaluate():
    df = load_dataset()
    
    feature_cols = ["review_text"] + NUMERICAL_COLS
    X = df[feature_cols]
    y = df["target"]

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Hybrid Preprocessor: Scaled Numerical + TF-IDF Text N-Grams
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=2000,
                    sublinear_tf=True,
                    stop_words="english",
                ),
                "review_text",
            ),
        ]
    )

    models = {
        "LogisticRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000, C=2.0)),
        ]),
        "SGDClassifier_LogLoss": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SGDClassifier(loss="log_loss", random_state=42, max_iter=1000)),
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=18, random_state=42, n_jobs=2)),
        ]),
    }

    results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_pipeline = None

    print("\n" + "=" * 50)
    print("STARTING HYBRID MODEL BENCHMARKING (STYLOMETRICS + TF-IDF)")
    print("=" * 50)

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
        }

        print(f"--> {name} Results:")
        print(f"    Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-Score : {f1:.4f}")
        print(f"    ROC-AUC  : {roc_auc:.4f}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_pipeline = pipeline

    print("\n" + "=" * 50)
    print(f"BEST MODEL: {best_model_name} (ROC-AUC: {best_roc_auc:.4f}, Accuracy: {results[best_model_name]['accuracy']*100:.2f}%)")
    print("=" * 50)

    # Save Best Model Pipeline
    pipeline_path = MODELS_DIR / "best_model_pipeline.joblib"
    joblib.dump({
        "pipeline": best_pipeline,
        "numerical_cols": NUMERICAL_COLS,
        "feature_names": feature_cols,
        "model_name": best_model_name,
        "metrics": results[best_model_name],
    }, pipeline_path)
    print(f"Saved best hybrid model pipeline to {pipeline_path}")

    # Save Metrics Report JSON
    metrics_path = MODELS_DIR / "metrics_report.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved model metrics report to {metrics_path}")

    # Feature Importance / Coefficients Visualization
    if best_model_name in ["LogisticRegression", "SGDClassifier_LogLoss"]:
        clf = best_pipeline.named_steps["classifier"]
        prep = best_pipeline.named_steps["preprocessor"]
        
        # Get feature names from preprocessor
        tf_vectorizer = prep.named_transformers_["text"]
        tfidf_feature_names = list(tf_vectorizer.get_feature_names_out())
        all_feature_names = NUMERICAL_COLS + tfidf_feature_names
        
        coefs = clf.coef_[0]
        top_positive_idx = np.argsort(coefs)[-15:]
        top_negative_idx = np.argsort(coefs)[:15]
        top_indices = np.hstack([top_negative_idx, top_positive_idx])
        
        plt.figure(figsize=(10, 7))
        plt.barh([all_feature_names[i] for i in top_indices], coefs[top_indices], color=["#EF4444" if coefs[i] > 0 else "#10B981" for i in top_indices])
        plt.title("Top Deceptive (Red) vs Authentic (Green) Hybrid Predictors")
        plt.xlabel("Coefficient Value")
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "feature_importance.png")
        plt.close()
        print(f"Saved hybrid feature importance plot to {MODELS_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    train_and_evaluate()
