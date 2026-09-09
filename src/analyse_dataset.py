"""
analyse_dataset.py

Phase 1: Dataset analysis.

Loads both train.ft.txt and test.ft.txt, prints actual statistics,
and saves simple EDA visualizations to ../visualizations/.

"""

import os
import sys

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_fasttext_file, print_load_summary, LABEL_NAMES
from preprocessing import clean_review

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np


# Paths 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH   = os.path.join(PROJECT_ROOT, "data", "train.ft.txt")
TEST_PATH    = os.path.join(PROJECT_ROOT, "data", "test.ft.txt")
VIZ_DIR      = os.path.join(PROJECT_ROOT, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)


def review_length_stats(texts: list, label: str = ""):
    """Print and return basic length statistics for a list of texts."""
    lengths = [len(t.split()) for t in texts]
    lengths_arr = np.array(lengths)
    stats = {
        "min":    int(lengths_arr.min()),
        "max":    int(lengths_arr.max()),
        "mean":   float(lengths_arr.mean()),
        "median": float(np.median(lengths_arr)),
        "p25":    float(np.percentile(lengths_arr, 25)),
        "p75":    float(np.percentile(lengths_arr, 75)),
    }
    if label:
        print(f"\n  Review-length statistics ({label}):")
        print(f"    Min    : {stats['min']} words")
        print(f"    Max    : {stats['max']} words")
        print(f"    Mean   : {stats['mean']:.1f} words")
        print(f"    Median : {stats['median']:.1f} words")
        print(f"    P25    : {stats['p25']:.1f} words")
        print(f"    P75    : {stats['p75']:.1f} words")
    return lengths_arr, stats


def plot_label_distribution(train_stats: dict, test_stats: dict, save_path: str):
    """Bar chart of positive/negative counts for train and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, stats, title in [
        (axes[0], train_stats, "Training Data"),
        (axes[1], test_stats,  "Test Data"),
    ]:
        counts = [stats["negative"], stats["positive"]]
        bars = ax.bar(
            ["Negative", "Positive"],
            counts,
            color=["#e05c5c", "#5c9ee0"],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Reviews")
        ax.set_xlabel("Sentiment Class")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_ylim(0, max(counts) * 1.15)

    plt.suptitle("Sentiment Class Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_length_distribution(train_lengths: np.ndarray, test_lengths: np.ndarray, save_path: str):
    """Histogram of review lengths (word count) for train and test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, lengths, title in [
        (axes[0], train_lengths, "Training Data"),
        (axes[1], test_lengths,  "Test Data"),
    ]:
        # Cap at 200 words for readability (most reviews are under that)
        clipped = np.clip(lengths, 0, 200)
        ax.hist(clipped, bins=50, color="#5c9ee0", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_title(f"Review Length Distribution\n({title})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Word Count (capped at 200)")
        ax.set_ylabel("Number of Reviews")
        ax.axvline(np.median(lengths), color="red", linestyle="--", linewidth=1.2, label=f"Median={np.median(lengths):.0f}")
        ax.axvline(np.mean(lengths),   color="orange", linestyle="--", linewidth=1.2, label=f"Mean={np.mean(lengths):.0f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def show_sample_reviews(texts: list, labels: list, n: int = 3):
    """Print a few sample reviews from each class."""
    print("\n  Sample Reviews:")
    for sentiment_val, sentiment_name in [(1, "POSITIVE"), (0, "NEGATIVE")]:
        indices = [i for i, l in enumerate(labels) if l == sentiment_val][:n]
        print(f"\n  --- {sentiment_name} ---")
        for idx in indices:
            preview = texts[idx][:120].replace("\n", " ")
            print(f"  [{idx}] {preview}...")


def main():
    print("\n" + "="*60)
    print("  Sentiment Analysis — Dataset Analysis")
    print("="*60)

    #  1. Load training data 
    print("\nLoading training data (this may take a minute for 1.5 GB)...")
    
    MAX_TRAIN = 600_000

    train_labels, train_texts, train_stats = load_fasttext_file(TRAIN_PATH, max_samples=MAX_TRAIN)
    print_load_summary("Training", train_stats)

    # 2. Load test data 
    print("Loading test data...")
    test_labels, test_texts, test_stats = load_fasttext_file(TEST_PATH)
    print_load_summary("Test", test_stats)

    # 3. Length statistics 
    print("\nComputing review-length statistics...")
    train_lengths, _ = review_length_stats(train_texts, "Train")
    test_lengths,  _ = review_length_stats(test_texts,  "Test")

    #  4. Sample reviews 
    show_sample_reviews(train_texts, train_labels)

    #  5. Preprocessing check on one example 
    example_raw = train_texts[0]
    example_clean = clean_review(example_raw)
    print(f"\n  Preprocessing example:")
    print(f"  Raw   : {example_raw[:100]}")
    print(f"  Cleaned: {example_clean[:100]}")

    # 6. Visualizations 
    print("\nGenerating visualizations...")
    plot_label_distribution(
        train_stats, test_stats,
        os.path.join(VIZ_DIR, "label_distribution.png"),
    )
    plot_length_distribution(
        train_lengths, test_lengths,
        os.path.join(VIZ_DIR, "length_distribution.png"),
    )

    print("\nDataset analysis complete.")

    # Return data for next phases
    return train_labels, train_texts, test_labels, test_texts


if __name__ == "__main__":
    main()
