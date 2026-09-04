"""
frontend/app.py
---------------
Streamlit frontend for Sentiment Analysis on Product Reviews.

Pages:
  1. Home          — project overview
  2. Analyse       — enter a review and get a prediction
  3. Dataset       — actual dataset statistics
  4. Model Results — actual test-set performance metrics

Run from the project root:
    streamlit run frontend/app.py
"""

import os
import sys
import json

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_sentiment

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "models", "results.json")
VIZ_DIR      = os.path.join(PROJECT_ROOT, "visualizations")


# ── Load results once ──────────────────────────────────────────────────────
@st.cache_data
def load_results():
    """Load training/evaluation results from disk (cached)."""
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis — Product Reviews",
    page_icon="📝",
    layout="centered",
)

# ── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Analyse Review", "📊 Dataset Insights", "📈 Model Performance"],
)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📝 Sentiment Analysis on Product Reviews")
    st.markdown("---")

    st.markdown(
        """
        ### Project Overview
        This application predicts whether a product review expresses a
        **Positive** or **Negative** sentiment using a trained machine-learning model.

        It was built as a B.Tech CSE academic project to demonstrate the full
        pipeline from raw text data through preprocessing, feature extraction,
        model training, and deployment.
        """
    )

    st.markdown("---")
    st.subheader("🎯 Objective")
    st.markdown(
        """
        > Build a complete, practical sentiment analysis system that can classify
        > product reviews as **Positive** or **Negative** in real time.
        """
    )

    st.markdown("---")
    st.subheader("📂 Dataset")
    st.markdown(
        """
        | Property | Value |
        |---|---|
        | Source | Kaggle — Amazon Reviews for Sentiment Analysis |
        | URL | https://www.kaggle.com/datasets/bittlingmayer/amazonreviews |
        | Format | fastText (`.ft.txt`) |
        | Classes | Binary: Positive / Negative |
        | Training records used | 600,000 |
        | Test records | 400,000 |

        > **Note:** The dataset is used solely as a source of product review
        > text for academic purposes. This application is not affiliated with
        > or endorsed by Amazon.
        """
    )

    st.markdown("---")
    st.subheader("🛠️ Technologies Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            - **Python 3.x**
            - **scikit-learn** — TF-IDF & ML models
            - **Pandas / NumPy** — data handling
            - **Matplotlib** — visualizations
            """
        )
    with col2:
        st.markdown(
            """
            - **FastAPI** — REST API backend
            - **Streamlit** — web frontend
            - **Joblib** — model persistence
            """
        )

    st.markdown("---")
    st.subheader("🔄 Project Workflow")
    st.markdown(
        """
        ```
        train.ft.txt + test.ft.txt
               ↓
        Data Loading & Parsing
               ↓
        Text Preprocessing
        (lowercase → strip HTML → normalise)
               ↓
        TF-IDF Feature Extraction
        (fit on training data only)
               ↓
        Model Training + Comparison
        (Logistic Regression vs Linear SVM)
               ↓
        Best Model Selected (Linear SVM)
               ↓
        Final Evaluation on Test Set
               ↓
        Saved ML Pipeline (.joblib)
               ↓
        FastAPI Backend  ←→  Streamlit Frontend
               ↓
        User enters review → Prediction returned
        ```
        """
    )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANALYSE REVIEW
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyse Review":
    st.title("🔍 Analyse a Product Review")
    st.markdown("Enter a product review below and click **Analyse Sentiment**.")
    st.markdown("---")

    review_text = st.text_area(
        label="Product Review",
        placeholder=(
            "Example: This product is amazing! Great quality and fast shipping. "
            "I would highly recommend it to everyone."
        ),
        height=150,
        max_chars=5000,
    )

    st.caption(f"Characters: {len(review_text)} / 5000")

    if st.button("🔎 Analyse Sentiment", type="primary"):
        # ── Input validation ──────────────────────────────────
        if not review_text.strip():
            st.error("Please enter a review before clicking Analyse.")
        elif len(review_text.strip()) < 3:
            st.warning("The review is too short. Please write at least a few words.")
        else:
            with st.spinner("Analysing..."):
                result = predict_sentiment(review_text)

            if result.get("error"):
                st.error(f"Could not classify: {result['error']}")
            else:
                st.markdown("---")
                sentiment = result["label"]

                if sentiment == "Positive":
                    st.success(f"### ✅ Sentiment: {sentiment}")
                    st.markdown(
                        """
                        <div style="background:#d4edda;padding:16px;border-radius:8px;
                                    border-left:6px solid #28a745;">
                            <h3 style="color:#155724;margin:0;">Positive Review Detected</h3>
                            <p style="color:#155724;margin:4px 0 0 0;">
                                The model predicts this review expresses positive sentiment.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"### ❌ Sentiment: {sentiment}")
                    st.markdown(
                        """
                        <div style="background:#f8d7da;padding:16px;border-radius:8px;
                                    border-left:6px solid #dc3545;">
                            <h3 style="color:#721c24;margin:0;">Negative Review Detected</h3>
                            <p style="color:#721c24;margin:4px 0 0 0;">
                                The model predicts this review expresses negative sentiment.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                with st.expander("ℹ️ How this works"):
                    st.markdown(
                        f"""
                        **Your review (preprocessed):**
                        > {result['cleaned'][:300]}{'...' if len(result['cleaned']) > 300 else ''}

                        **Steps taken:**
                        1. Your text was converted to lowercase.
                        2. HTML tags and special characters were removed.
                        3. The cleaned text was converted to a TF-IDF feature vector.
                        4. The trained Linear SVM model classified it as **{sentiment}**.

                        > Note: Linear SVM does not produce probability scores,
                        > so no confidence percentage is shown.
                        """
                    )

    st.markdown("---")
    st.markdown("**Try these examples:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            *Positive example:*
            > "Works exactly as described. Arrived quickly and the
            > build quality is excellent. Very happy with this purchase!"
            """
        )
    with col2:
        st.markdown(
            """
            *Negative example:*
            > "Completely disappointed. The product broke within a week
            > and customer service was no help at all. Do not buy this."
            """
        )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DATASET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Insights":
    st.title("📊 Dataset Insights")
    st.markdown("All statistics are calculated from the actual dataset files.")
    st.markdown("---")

    results = load_results()
    if results is None:
        st.error("results.json not found. Please run `python src/train_model.py` first.")
        st.stop()

    ds = results["dataset"]

    # ── Summary table ─────────────────────────────────────────────
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Reviews", f"{ds['train_total']:,}")
    col2.metric("Test Reviews",     f"{ds['test_total']:,}")
    col3.metric("Total",            f"{ds['train_total'] + ds['test_total']:,}")

    st.markdown("---")

    # ── Class distribution table ──────────────────────────────────
    st.subheader("Sentiment Class Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training Data**")
        st.markdown(
            f"""
            | Class | Count | % |
            |---|---|---|
            | ✅ Positive | {ds['train_positive']:,} | {ds['train_positive']/ds['train_total']*100:.1f}% |
            | ❌ Negative | {ds['train_negative']:,} | {ds['train_negative']/ds['train_total']*100:.1f}% |
            """
        )

    with col2:
        st.markdown("**Test Data**")
        st.markdown(
            f"""
            | Class | Count | % |
            |---|---|---|
            | ✅ Positive | {ds['test_positive']:,} | {ds['test_positive']/ds['test_total']*100:.1f}% |
            | ❌ Negative | {ds['test_negative']:,} | {ds['test_negative']/ds['test_total']*100:.1f}% |
            """
        )

    st.markdown("---")

    # ── Bar chart (matplotlib, rendered as image) ──────────────────
    st.subheader("Visualisations")
    label_img = os.path.join(VIZ_DIR, "label_distribution.png")
    length_img = os.path.join(VIZ_DIR, "length_distribution.png")

    if os.path.exists(label_img):
        st.image(label_img, caption="Positive vs Negative Distribution", use_container_width=True)
    else:
        st.info("Run `python src/analyse_dataset.py` to generate visualisations.")

    if os.path.exists(length_img):
        st.image(length_img, caption="Review Length Distribution", use_container_width=True)

    st.markdown("---")
    st.subheader("About the Dataset Format")
    st.markdown(
        """
        The dataset uses the **fastText format**: one review per line,
        with the sentiment label at the start.

        ```
        __label__2 Great product, works perfectly and arrived on time!
        __label__1 Terrible quality, broke within a week.
        ```

        | Label | Meaning | Original Stars |
        |---|---|---|
        | `__label__1` | Negative | 1-star and 2-star reviews |
        | `__label__2` | Positive | 4-star and 5-star reviews |

        > 3-star (neutral) reviews are **not included** in the dataset,
        > making this a binary classification problem.
        """
    )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown(
        "All metrics are calculated from the **actual test set** (400,000 reviews). "
        "No values are fabricated."
    )
    st.markdown("---")

    results = load_results()
    if results is None:
        st.error("results.json not found. Please run `python src/train_model.py` first.")
        st.stop()

    test   = results["test_results"]
    val    = results["validation"]
    lr_val = val["logistic_regression"]
    sv_val = val["linear_svm"]

    # ── Model comparison (validation) ─────────────────────────────
    st.subheader("Model Comparison — Validation Set")
    st.caption("Validation split: 20% of training data (120,000 reviews).")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown("**Logistic Regression**")
        st.markdown(
            f"""
            | Metric | Score |
            |---|---|
            | Accuracy  | {lr_val['accuracy']:.4f} |
            | Precision | {lr_val['precision']:.4f} |
            | Recall    | {lr_val['recall']:.4f} |
            | F1-Score  | {lr_val['f1']:.4f} |
            """
        )

    with comp_col2:
        st.markdown("**Linear SVM** ✅ *(selected)*")
        st.markdown(
            f"""
            | Metric | Score |
            |---|---|
            | Accuracy  | {sv_val['accuracy']:.4f} |
            | Precision | {sv_val['precision']:.4f} |
            | Recall    | {sv_val['recall']:.4f} |
            | F1-Score  | {sv_val['f1']:.4f} |
            """
        )

    st.info(
        f"**Linear SVM** was selected because it achieved a higher validation "
        f"F1-score ({sv_val['f1']:.4f}) compared to Logistic Regression ({lr_val['f1']:.4f})."
    )

    st.markdown("---")

    # ── Final test results ─────────────────────────────────────────
    st.subheader("Final Test Results — Linear SVM")
    st.caption("Evaluated on the untouched test set (400,000 reviews).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{test['accuracy']:.4f}")
    m2.metric("Precision", f"{test['precision']:.4f}")
    m3.metric("Recall",    f"{test['recall']:.4f}")
    m4.metric("F1-Score",  f"{test['f1']:.4f}")

    st.markdown("---")

    # ── Confusion matrix ───────────────────────────────────────────
    st.subheader("Confusion Matrix")
    cm = test["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes = ["Negative", "Positive"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix — Test Set")

    thresh = np.array(cm).max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i][j]:,}",
                ha="center", va="center",
                color="white" if cm[i][j] > thresh else "black",
                fontsize=12,
            )

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        | | Predicted Negative | Predicted Positive |
        |---|---|---|
        | **Actual Negative** | True Negative (TN) | False Positive (FP) |
        | **Actual Positive** | False Negative (FN) | True Positive (TP) |
        """
    )

    st.markdown("---")

    # ── Configuration used ─────────────────────────────────────────
    st.subheader("Training Configuration")
    st.markdown(
        f"""
        | Parameter | Value |
        |---|---|
        | Training records used | {results['train_samples_used']:,} |
        | Validation split | {int(results['val_split']*100)}% of training data |
        | TF-IDF max features | {results['tfidf_max_features']:,} |
        | TF-IDF n-gram range | {results['tfidf_ngram_range'][0]}–{results['tfidf_ngram_range'][1]} |
        | Best model | {results['model_used']} |
        | Random seed | {results['random_seed']} |
        """
    )

    st.markdown("---")
    st.subheader("⚠️ Limitations")
    st.markdown(
        """
        - The model handles only **binary sentiment** (Positive / Negative).
          There is no neutral class.
        - Sarcasm, mixed opinions, and regional slang may be misclassified.
        - Performance depends on the training dataset distribution.
        - The system is not connected to any live review platform.
        - Very short reviews (fewer than 3 characters) are rejected.
        """
    )
