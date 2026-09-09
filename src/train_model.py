"""
train_model.py

Phases 3-7: Preprocessing → TF-IDF → Train → Compare → Save

What this script does, step by step:
1. Load 600,000 training reviews from train.ft.txt
2. Preprocess all review texts (lowercase, strip HTML, etc.)
3. Split into train/validation (80/20) from training data ONLY
4. Build a TF-IDF vectorizer and fit it on the training split ONLY
5. Transform train and validation splits
6. Train Logistic Regression and Linear SVM on the training split
7. Evaluate both models on the validation split
8. Select the best model
9. Load full test data (test.ft.txt) — kept untouched until now
10. Evaluate the best model on the test split
11. Save the fitted TF-IDF + best model as a single pipeline
12. Save evaluation results to models/results.json

Run from the project root:
    python src/train_model.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from data_loader   import load_fasttext_file, print_load_summary
from preprocessing import preprocess_texts

from sklearn.model_selection      import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model         import LogisticRegression
from sklearn.svm                  import LinearSVC
from sklearn.pipeline             import Pipeline
from sklearn.metrics              import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import joblib
import numpy as np


# Paths 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH   = os.path.join(PROJECT_ROOT, "data", "train.ft.txt")
TEST_PATH    = os.path.join(PROJECT_ROOT, "data", "test.ft.txt")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PIPELINE_PATH = os.path.join(MODELS_DIR, "sentiment_pipeline.joblib")
RESULTS_PATH  = os.path.join(MODELS_DIR, "results.json")


#  Configuration 
# How many training records to use.
# The full training file has ~3.6 M reviews; 600k gives a good
# accuracy/speed balance on a typical laptop (≈5–10 min training).
MAX_TRAIN_SAMPLES = 600_000

# Validation split ratio (from training data only)
VAL_SPLIT  = 0.20
RANDOM_SEED = 42


def evaluate(model, X, y_true, split_name: str) -> dict:
    """
    Evaluate a model and print a clean report.
    Returns a dict with the key metrics.
    """
    y_pred = model.predict(X)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary")
    rec  = recall_score(y_true,    y_pred, average="binary")
    f1   = f1_score(y_true,        y_pred, average="binary")
    cm   = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n  {split_name} Results:")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Neg_pred  Pos_pred")
    print(f"      Neg_actual  {cm[0][0]:>7,}  {cm[0][1]:>7,}")
    print(f"      Pos_actual  {cm[1][0]:>7,}  {cm[1][1]:>7,}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


def main():
    print("\n" + "="*60)
    print("  Sentiment Analysis — Model Training")
    print("="*60)

    #  Step 1: Load training data 
    print(f"\nStep 1: Loading {MAX_TRAIN_SAMPLES:,} training reviews...")
    t0 = time.time()
    train_labels, train_texts_raw, train_stats = load_fasttext_file(TRAIN_PATH, max_samples=MAX_TRAIN_SAMPLES)
    print_load_summary("Training", train_stats)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    #  Step 2: Preprocessing 
    print("\nStep 2: Preprocessing training texts...")
    t0 = time.time()
    train_texts_clean = preprocess_texts(train_texts_raw)
    print(f"  Preprocessed {len(train_texts_clean):,} reviews in {time.time()-t0:.1f}s")

    # Step 3: Train / Validation split 
    print(f"\nStep 3: Splitting into train ({int((1-VAL_SPLIT)*100)}%) / validation ({int(VAL_SPLIT*100)}%)...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        train_texts_clean,
        train_labels,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=train_labels,  # keep class balance in both splits
    )
    print(f"  Training split   : {len(X_train_raw):,} reviews")
    print(f"  Validation split : {len(X_val_raw):,} reviews")

    #  Step 4: TF-IDF Vectorizer 
    # Parameters explained:
    #   ngram_range=(1,2) : use single words AND two-word phrases
    #                        so "not good" is treated as one feature
    #   min_df=5          : ignore terms that appear in fewer than 5 docs
    #                        (removes typos and very rare tokens)
    #   max_df=0.95       : ignore terms in >95% of docs (stop-words effect)
    #   max_features=150000: keep the 150,000 most informative features
    #                        to control memory and speed
    #   sublinear_tf=True : apply log(1+tf) — reduces impact of very
    #                        frequent words within one document
    print("\nStep 4: Fitting TF-IDF vectorizer on training split only...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=150_000,
        sublinear_tf=True,
    )
    X_train = tfidf.fit_transform(X_train_raw)   # FIT + TRANSFORM on train
    X_val   = tfidf.transform(X_val_raw)         # TRANSFORM only on val
    print(f"  Vocabulary size  : {len(tfidf.vocabulary_):,} features")
    print(f"  Train matrix     : {X_train.shape}")
    print(f"  Val matrix       : {X_val.shape}")
    print(f"  TF-IDF done in   : {time.time()-t0:.1f}s")

    #  Step 5: Train Model 1 — Logistic Regression 
    print("\nStep 5a: Training Logistic Regression...")
    # C=1.0 is the regularisation strength (inverse); solver='saga' is
    # efficient for large sparse datasets; max_iter=1000 ensures convergence.
    t0 = time.time()
    lr_model = LogisticRegression(C=1.0, solver="saga", max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    lr_val_results = evaluate(lr_model, X_val, y_val, "Logistic Regression — Validation")

    #  Step 6: Train Model 2 — Linear SVM 
    print("\nStep 5b: Training Linear SVM...")
    # LinearSVC is fast for high-dimensional text; C=0.1 gives slight
    # regularisation that works well for text classification.
    t0 = time.time()
    svm_model = LinearSVC(C=0.1, max_iter=2000, random_state=RANDOM_SEED)
    svm_model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    svm_val_results = evaluate(svm_model, X_val, y_val, "Linear SVM — Validation")

    #  Step 7: Model Selection 
    print("\nStep 6: Selecting best model based on validation F1-score...")
    if svm_val_results["f1"] >= lr_val_results["f1"]:
        best_model      = svm_model
        best_model_name = "Linear SVM"
        best_val        = svm_val_results
    else:
        best_model      = lr_model
        best_model_name = "Logistic Regression"
        best_val        = lr_val_results

    print(f"  Selected: {best_model_name} (Val F1: {best_val['f1']:.4f})")

    # Step 8: Load & preprocess test data 
    print("\nStep 7: Loading test data (untouched until now)...")
    t0 = time.time()
    test_labels, test_texts_raw, test_stats = load_fasttext_file(TEST_PATH)
    print_load_summary("Test", test_stats)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("  Preprocessing test texts...")
    test_texts_clean = preprocess_texts(test_texts_raw)

    # Transform using the ALREADY-FITTED TF-IDF (no re-fitting!)
    print("  Transforming test texts with fitted TF-IDF...")
    X_test = tfidf.transform(test_texts_clean)
    print(f"  Test matrix: {X_test.shape}")

    # Step 9: Final test evaluation 
    print(f"\nStep 8: Final evaluation of '{best_model_name}' on test data...")
    test_results = evaluate(best_model, X_test, test_labels, f"{best_model_name} — TEST SET")

    print("\n  Full classification report:")
    y_test_pred = best_model.predict(X_test)
    print(classification_report(test_labels, y_test_pred, target_names=["Negative", "Positive"]))

    #  Step 10: Save pipeline 
    # We save a sklearn Pipeline object that wraps both the fitted
    # TF-IDF and the model. When loaded later, calling pipeline.predict()
    # automatically preprocesses → vectorizes → predicts.
    # Note: preprocessing is NOT inside the sklearn pipeline because we
    # want to handle it in Python before passing to the pipeline.
    # The pipeline here = (tfidf_step, model_step).
    print(f"\nStep 9: Saving pipeline to {PIPELINE_PATH}...")
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("model", best_model),
    ])
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"  Saved.")

    # Step 11: Save results JSON 
    results = {
        "model_used":         best_model_name,
        "train_samples_used": MAX_TRAIN_SAMPLES,
        "val_split":          VAL_SPLIT,
        "random_seed":        RANDOM_SEED,
        "tfidf_max_features": 150_000,
        "tfidf_ngram_range":  [1, 2],
        "validation": {
            "logistic_regression": lr_val_results,
            "linear_svm":          svm_val_results,
        },
        "test_results": test_results,
        "dataset": {
            "train_positive": train_stats["positive"],
            "train_negative": train_stats["negative"],
            "train_total":    train_stats["valid_records"],
            "test_positive":  test_stats["positive"],
            "test_negative":  test_stats["negative"],
            "test_total":     test_stats["valid_records"],
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_PATH}")

    print("\n" + "="*60)
    print(f"  TRAINING COMPLETE")
    print(f"  Best model : {best_model_name}")
    print(f"  Test Acc.  : {test_results['accuracy']:.4f}")
    print(f"  Test F1    : {test_results['f1']:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
