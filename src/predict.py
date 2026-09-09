"""
predict.py

Loads the saved ML pipeline and predicts sentiment for a given review.

The pipeline contains:
  1. Fitted TF-IDF vectorizer
  2. Trained Linear SVM model

Usage (from project root):
    python src/predict.py "This product is amazing!"
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_review


PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_pipeline.joblib")

LABEL_NAMES = {0: "Negative", 1: "Positive"}

# Module-level cache: load the pipeline once, reuse for all calls.
_pipeline = None


def load_pipeline():
    """Load the saved pipeline from disk (cached after first call)."""
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(
                f"Pipeline not found at {PIPELINE_PATH}. "
                "Run 'python src/train_model.py' first."
            )
        _pipeline = joblib.load(PIPELINE_PATH)
    return _pipeline


def predict_sentiment(review_text: str) -> dict:
    """
    Predict the sentiment of a single review.

    Parameters
    ----------
    review_text : raw review string from the user

    Returns
    -------
    dict with:
      "label"      : "Positive" or "Negative"
      "label_int"  : 1 or 0
      "confidence" : None  (LinearSVC does not produce calibrated probabilities)
      "cleaned"    : the preprocessed text that was fed to the model
    """
    pipeline = load_pipeline()

    # Preprocess the text the same way it was preprocessed during training
    cleaned = clean_review(review_text)

    if not cleaned:
        return {
            "label":      None,
            "label_int":  None,
            "confidence": None,
            "cleaned":    cleaned,
            "error":      "Review is empty after preprocessing.",
        }

    # Pipeline expects a list of strings
    prediction = pipeline.predict([cleaned])[0]

    return {
        "label":      LABEL_NAMES[prediction],
        "label_int":  int(prediction),
        "confidence": None,   # LinearSVC does not give reliable probabilities
        "cleaned":    cleaned,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"Your review text here\"")
        sys.exit(1)

    review = " ".join(sys.argv[1:])
    result = predict_sentiment(review)
    print(f"\nReview   : {review}")
    print(f"Cleaned  : {result['cleaned']}")
    print(f"Sentiment: {result['label']}")
