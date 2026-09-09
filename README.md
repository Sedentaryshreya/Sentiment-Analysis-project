# Sentiment Analysis on Product Reviews Using Python

A B.Tech CSE academic project that predicts whether a product review expresses **Positive** or **Negative** sentiment using traditional machine learning.

---

## Project Overview

This project builds a complete sentiment analysis pipeline from raw text data through preprocessing, TF-IDF feature extraction, machine learning model training, REST API deployment, and a web-based user interface.

A user enters a product review. The system classifies it as **Positive** or **Negative** in real time using a trained Linear SVM model.

---

## Problem Statement

Online product reviews contain valuable customer feedback, but manually reading thousands of reviews is impractical. Automatically classifying review sentiment helps businesses and consumers quickly understand the general opinion about a product.

---

## Objective

Build a practical, end-to-end sentiment analysis system that:
- Trains on real product review data
- Classifies new reviews as Positive or Negative
- Exposes predictions through a REST API
- Displays results in a clean web interface

---

## Dataset

| Property | Details |
|---|---|
| Name | Amazon Reviews for Sentiment Analysis |
| Source | Kaggle |
| URL | https://www.kaggle.com/datasets/bittlingmayer/amazonreviews |
| Format | fastText (`.ft.txt`) |
| Classes | Binary — Positive / Negative |
| Training file | `train.ft.txt` |
| Test file | `test.ft.txt` |

> The dataset is used solely as a source of product review text for academic purposes. This project is not affiliated with or endorsed by Amazon.

### Dataset Format

Each line contains one review in fastText format:

```
__label__2 Great product, arrived quickly and works perfectly!
__label__1 Terrible quality, broke within a week.
```

| Label | Meaning | Original Stars |
|---|---|---|
| `__label__1` | Negative sentiment | 1-star and 2-star reviews |
| `__label__2` | Positive sentiment | 4-star and 5-star reviews |

3-star (neutral) reviews are not included, making this a **binary classification** problem.

### Records Used

| Split | Records | Positive | Negative |
|---|---|---|---|
| Training (used) | 600,000 | 303,665 | 296,335 |
| Test (untouched) | 400,000 | 200,000 | 200,000 |

> The full training file contains approximately 3.6 million records. 600,000 were used to balance performance and training time on standard hardware.

---

## Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.x |
| ML / NLP | scikit-learn (TF-IDF, Logistic Regression, LinearSVC) |
| Data handling | NumPy, Pandas |
| Visualizations | Matplotlib |
| API backend | FastAPI + Uvicorn |
| Web frontend | Streamlit |
| Model saving | Joblib |

---

## System Architecture

```
train.ft.txt + test.ft.txt
         |
    Data Loading (data_loader.py)
         |
  Text Preprocessing (preprocessing.py)
         |
  TF-IDF Feature Extraction
  [fit on training data ONLY]
         |
  Model Training + Comparison
  (Logistic Regression vs Linear SVM)
  [validation split from training data only]
         |
  Best Model Selected (Linear SVM)
         |
  Final Evaluation on test.ft.txt
         |
  Saved Pipeline (sentiment_pipeline.joblib)
         |
  FastAPI Backend (backend/main.py)
         |
  Streamlit Frontend (frontend/app.py)
         |
  User enters review → Prediction shown
```

---

## Project Structure

```
sentiment-analysis-project/
│
├── data/
│   ├── train.ft.txt          # Training data (fastText format)
│   └── test.ft.txt           # Test data (fastText format)
│
├── src/
│   ├── data_loader.py        # Parses fastText files
│   ├── preprocessing.py      # Text cleaning functions
│   ├── train_model.py        # Full training pipeline
│   ├── analyse_dataset.py    # EDA and visualisations
│   └── predict.py            # Load pipeline, predict one review
│
├── backend/
│   └── main.py               # FastAPI REST API
│
├── frontend/
│   └── app.py                # Streamlit web interface
│
├── models/
│   ├── sentiment_pipeline.joblib   # Saved TF-IDF + model
│   └── results.json                # Actual evaluation results
│
├── visualizations/
│   ├── label_distribution.png
│   └── length_distribution.png
│
├── integration_test.py       # End-to-end test suite
├── test_backend.py           # Backend-only tests
├── requirements.txt
└── README.md
```

---

## Data Preprocessing

Applied in `src/preprocessing.py` using `clean_review()`:

| Step | Action | Reason |
|---|---|---|
| 1 | Lowercase | Case does not carry sentiment |
| 2 | Remove HTML tags | Some reviews contain `<br />` etc. |
| 3 | Remove non-alphanumeric characters (keep apostrophes) | Clean noise; keep contractions like "don't" |
| 4 | Collapse multiple spaces | Normalise whitespace |
| 5 | Strip edges | Remove leading/trailing spaces |

Negations (`not`, `never`, `no`) are **preserved** because they carry strong sentiment signal. The same function is used at training time and prediction time.

---

## TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numbers.

| Parameter | Value | Reason |
|---|---|---|
| `ngram_range` | (1, 2) | Captures single words and two-word phrases like "not good" |
| `min_df` | 5 | Ignores words appearing in fewer than 5 reviews (typos/noise) |
| `max_df` | 0.95 | Ignores words in more than 95% of reviews (common stop words) |
| `max_features` | 150,000 | Keeps the 150k most informative features |
| `sublinear_tf` | True | Applies log(1+tf) to reduce impact of very frequent words |

**Important:** The vectorizer is fit **only on training data** and then used to transform both training and test data. The test set is never seen during fitting.

---

## Machine Learning Models

### Model 1 — Logistic Regression
- Solver: SAGA (efficient for large sparse datasets)
- C: 1.0 (regularisation strength)
- Finds the probability boundary between Positive and Negative

### Model 2 — Linear SVM (LinearSVC)
- C: 0.1 (stronger regularisation — works well for text)
- Finds the maximum-margin hyperplane separating classes
- Faster and often more accurate than Logistic Regression for text

Both models were trained on the 80% training split and compared on the 20% validation split. The better model was then evaluated once on the untouched test set.

---

## Model Evaluation

### Validation Set Results (120,000 reviews)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9309 | 0.9308 | 0.9328 | 0.9318 |
| **Linear SVM** ✅ | **0.9328** | **0.9329** | **0.9345** | **0.9337** |

Linear SVM was selected (higher F1-score).

### Final Test Results — Linear SVM (400,000 reviews)

| Metric | Score |
|---|---|
| **Accuracy** | **0.9325** |
| **Precision** | **0.9323** |
| **Recall** | **0.9327** |
| **F1-Score** | **0.9325** |

### Confusion Matrix (Test Set)

```
                  Predicted Negative   Predicted Positive
Actual Negative       186,450               13,550
Actual Positive        13,466              186,534
```

> All results are from actual execution. No values are fabricated.

---

## Sample Predictions

| Review | Predicted |
|---|---|
| "Amazing product! Works perfectly and arrived fast." | ✅ Positive |
| "Complete garbage. Stopped working after 2 days." | ❌ Negative |
| "Not good at all, I would never recommend this." | ❌ Negative |
| "Works exactly as described. Very happy with it." | ✅ Positive |
| "Do not buy this. Very disappointing purchase." | ❌ Negative |

---

## Installation

```bash
# 1. Clone or download the project
cd sentiment-analysis-project

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place data files in data/
# data/train.ft.txt
# data/test.ft.txt
```

---

## Running the Project

### Step 1 — Analyse Dataset (optional)
```bash
python src/analyse_dataset.py
```
Prints dataset statistics and saves visualisations to `visualizations/`.

### Step 2 — Train the Model
```bash
python src/train_model.py
```
Trains both models, evaluates, selects the best, and saves:
- `models/sentiment_pipeline.joblib`
- `models/results.json`

This step takes approximately **5–10 minutes** on standard hardware.

### Step 3 — Run Integration Tests
```bash
python integration_test.py
```

### Step 4 — Start the FastAPI Backend

Open a **terminal** and run:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Test it:
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"review\": \"This product is amazing!\"}"
```

API documentation is automatically available at: http://localhost:8000/docs

### Step 5 — Start the Streamlit Frontend

Open a **separate terminal** and run:
```bash
streamlit run frontend/app.py
```

The browser will open at: http://localhost:8501

> The Streamlit frontend calls the ML model **directly** (not via the FastAPI backend), so it works without the backend running. The backend is a standalone API for integration with other tools.

---

## Limitations

- The model handles only **binary sentiment** (Positive / Negative). There is no neutral class.
- The dataset does not contain 3-star reviews, so middle-of-the-road reviews may be misclassified.
- Sarcasm, mixed opinions, domain-specific slang, and spelling errors may reduce accuracy.
- The system is not connected to any live product review platform.
- Linear SVM does not provide calibrated probability scores.
- Performance depends on the training data distribution.

---

## Future Scope

- Add a neutral class using an appropriate three-class dataset
- Try advanced models: BERT, RoBERTa, or other transformer-based models
- Multilingual sentiment analysis
- Aspect-based sentiment analysis (e.g., "battery is bad but display is great")
- Better handling of sarcasm and negation with context-aware models
- Deployment on a cloud platform (AWS, GCP, Azure, Heroku)
- Analysis of reviews across additional product categories

---

## Dataset Attribution

Dataset: **Amazon Reviews for Sentiment Analysis**  
Available at: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews  
Used for academic purposes only.

---

*Project Title: Sentiment Analysis on Product Reviews Using Python*  
*Type: B.Tech CSE Academic Project*
 ## *Author: Shreya Gautam*
