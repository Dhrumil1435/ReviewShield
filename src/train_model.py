"""
train_model.py
Trains, evaluates, and serializes machine learning models for ReviewShield deceptive review detection.
Models evaluated: Logistic Regression, Random Forest, HistGradientBoosting.
Saves the best performing pipeline to models/best_model_pipeline.joblib.
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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from src.config import get_db_engine, MODELS_DIR

FEATURE_COLS = [
    "punctuation_freq",
    "vocab_diversity",
    "readability_score",
    "avg_sentence_length",
    "sentiment_score",
    "rating_sentiment_gap",
]


def load_dataset():
    engine = get_db_engine()
    query = """
        SELECT f.review_id, r.label, f.punctuation_freq, f.vocab_diversity, 
               f.readability_score, f.avg_sentence_length, f.sentiment_score, f.rating_sentiment_gap
        FROM engineered_features f
        JOIN raw_reviews r ON f.review_id = r.review_id
    """
    df = pd.read_sql(query, engine)
    
    # Encode Target: CG (Computer Generated / Deceptive) = 1, OR (Original / Genuine) = 0
    df["target"] = df["label"].apply(lambda x: 1 if str(x).strip().upper() in ["CG", "1", "FAKE"] else 0)
    
    print(f"Dataset loaded successfully: {len(df)} total reviews.")
    print(f"Target distribution (1=Deceptive, 0=Genuine):\n{df['target'].value_counts()}")
    return df


def train_and_evaluate():
    df = load_dataset()
    X = df[FEATURE_COLS]
    y = df["target"]

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=500)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42, n_jobs=2)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", HistGradientBoostingClassifier(max_iter=50, random_state=42)),
        ]),
    }

    results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_pipeline = None

    print("\n" + "=" * 50)
    print("STARTING MODEL BENCHMARKING")
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
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-Score : {f1:.4f}")
        print(f"    ROC-AUC  : {roc_auc:.4f}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_pipeline = pipeline

    print("\n" + "=" * 50)
    print(f"BEST MODEL: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    print("=" * 50)

    # Save Best Model Pipeline
    pipeline_path = MODELS_DIR / "best_model_pipeline.joblib"
    joblib.dump({
        "pipeline": best_pipeline,
        "feature_names": FEATURE_COLS,
        "model_name": best_model_name,
        "metrics": results[best_model_name],
    }, pipeline_path)
    print(f"Saved best model pipeline to {pipeline_path}")

    # Save Metrics Report JSON
    metrics_path = MODELS_DIR / "metrics_report.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved model metrics report to {metrics_path}")

    # Feature Importance Plot
    if "RandomForest" in models:
        rf_model = models["RandomForest"].named_steps["classifier"]
        importances = rf_model.feature_importances_
        plt.figure(figsize=(8, 5))
        plt.barh(FEATURE_COLS, importances, color="#4F46E5")
        plt.title("Random Forest Feature Importances")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "feature_importance.png")
        plt.close()
        print(f"Saved feature importance plot to {MODELS_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    train_and_evaluate()
