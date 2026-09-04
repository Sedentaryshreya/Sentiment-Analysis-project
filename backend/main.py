"""
backend/main.py
---------------
FastAPI backend for the Sentiment Analysis project.

Endpoints:
  GET  /health   — check that the server is running
  POST /predict  — predict sentiment of a product review

Run from the project root:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Or:
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from predict import predict_sentiment, load_pipeline


# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "Predicts whether a product review is Positive or Negative. "
        "Model: TF-IDF + Linear SVM trained on a large review dataset."
    ),
    version="1.0.0",
)

# Pre-load the pipeline when the server starts so the first request isn't slow.
@app.on_event("startup")
def startup_event():
    load_pipeline()
    print("ML pipeline loaded and ready.")


# ── Request / Response schemas ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    review: str

    @field_validator("review")
    @classmethod
    def review_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Review text cannot be empty.")
        return v.strip()


class PredictResponse(BaseModel):
    sentiment:   str           # "Positive" or "Negative"
    confidence:  float | None  # None for LinearSVC (no calibrated probability)
    review_used: str           # the preprocessed text that was classified


class HealthResponse(BaseModel):
    status:  str
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check():
    """Returns OK when the backend is running and the model is loaded."""
    return HealthResponse(status="ok", message="Sentiment Analysis API is running.")


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest):
    """
    Predict sentiment for a product review.

    - **review**: text of the product review (1 – 5,000 characters)

    Returns:
    - **sentiment**: "Positive" or "Negative"
    - **confidence**: null (LinearSVC does not produce calibrated probabilities)
    - **review_used**: the cleaned version of the review that was classified
    """
    # Length guard: very short reviews are hard to classify meaningfully.
    if len(request.review) < 3:
        raise HTTPException(
            status_code=422,
            detail="Review is too short to classify. Please provide more text.",
        )

    # Length guard: cap at 5,000 characters to avoid abuse.
    if len(request.review) > 5_000:
        raise HTTPException(
            status_code=422,
            detail="Review exceeds the 5,000-character limit. Please shorten it.",
        )

    result = predict_sentiment(request.review)

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    return PredictResponse(
        sentiment   = result["label"],
        confidence  = result["confidence"],
        review_used = result["cleaned"],
    )
