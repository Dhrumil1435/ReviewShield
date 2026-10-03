"""
api.py
FastAPI Production REST API for ReviewShield.
Provides programmatically queryable endpoints for real-time deceptive review detection,
word-level Explainable AI (XAI) attributions, and bulk batch processing.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.inference import ReviewAnalyzer

# Initialize FastAPI App
app = FastAPI(
    title="ReviewShield REST API",
    description=(
        "Production REST API for deceptive review detection powered by "
        "TF-IDF N-Grams, Stylometric Profiling, VADER sentiment analysis, "
        "and Meta's RoBERTa deep learning contextual sentiment engine."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for E-Commerce Platforms & Web Frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Analyzer Singleton Instance
analyzer_instance: Optional[ReviewAnalyzer] = None


def get_analyzer() -> ReviewAnalyzer:
    global analyzer_instance
    if analyzer_instance is None:
        analyzer_instance = ReviewAnalyzer()
    return analyzer_instance


# Pydantic Schemas
class SingleReviewRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=3,
        description="The full text content of the product or service review.",
        example="This product is absolutely amazing! Outstanding build quality and fast shipping.",
    )
    rating: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="The numerical star rating assigned by the reviewer (1.0 to 5.0).",
        example=5.0,
    )


class BatchReviewRequest(BaseModel):
    reviews: List[SingleReviewRequest] = Field(
        ...,
        min_items=1,
        max_items=500,
        description="List of review items to process in a single batch request.",
    )


class FeatureBreakdown(BaseModel):
    punctuation_freq: float
    vocab_diversity: float
    readability_score: float
    avg_sentence_length: float
    sentiment_score: float
    rating_sentiment_gap: float
    roberta_sentiment: float
    roberta_rating_sentiment_gap: float
    vader_roberta_dissonance: float


class PredictionResponse(BaseModel):
    review_text: str
    rating: float
    is_deceptive: bool
    classification: str
    deceptive_probability: float
    deceptive_percentage: float
    risk_factors: List[str]
    explained_html: str
    features: FeatureBreakdown


class BatchPredictionResponse(BaseModel):
    total_scanned: int
    deceptive_count: int
    genuine_count: int
    deceptive_ratio_percentage: float
    results: List[PredictionResponse]


# API Routes
@app.get("/", tags=["Health & Info"])
def root_info():
    """Welcome endpoint providing service metadata and documentation links."""
    return {
        "service": "ReviewShield REST API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "interactive_redoc": "/redoc",
    }


@app.get("/health", tags=["Health & Info"])
def health_check():
    """System health check endpoint verifying ML model pipeline loading status."""
    try:
        analyzer = get_analyzer()
        model_loaded = analyzer.pipeline is not None
        return {
            "status": "healthy" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "model_path": str(analyzer.model_path),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failure: {str(e)}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Deception Detection"])
def predict_single_review(request: SingleReviewRequest):
    """
    Evaluates a single review for deceptive intent.
    Returns deceptive probability score, risk signals, XAI word attributions, and stylometric features.
    """
    try:
        analyzer = get_analyzer()
        res = analyzer.analyze(request.review_text, request.rating)
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Deception Detection"])
def predict_batch_reviews(request: BatchReviewRequest):
    """
    Processes a list of review objects in bulk.
    Returns aggregated metrics along with individual prediction breakdowns.
    """
    try:
        analyzer = get_analyzer()
        results = []
        deceptive_count = 0

        for item in request.reviews:
            res = analyzer.analyze(item.review_text, item.rating)
            if res["is_deceptive"]:
                deceptive_count += 1
            results.append(res)

        total_scanned = len(results)
        genuine_count = total_scanned - deceptive_count
        ratio_pct = round((deceptive_count / total_scanned) * 100, 2) if total_scanned > 0 else 0.0

        return {
            "total_scanned": total_scanned,
            "deceptive_count": deceptive_count,
            "genuine_count": genuine_count,
            "deceptive_ratio_percentage": ratio_pct,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
