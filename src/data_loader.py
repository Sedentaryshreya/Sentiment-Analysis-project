"""
data_loader.py

Loads and parses the fastText-format review files.

fastText format (one review per line):
    __label__1 This product is terrible.
    __label__2 Loved it, great quality!

Label mapping:
    __label__1 -> 0  (Negative: 1-star and 2-star reviews)
    __label__2 -> 1  (Positive: 4-star and 5-star reviews)
"""

import os


LABEL_MAP = {
    "__label__1": 0,   # Negative
    "__label__2": 1,   # Positive
}

LABEL_NAMES = {0: "Negative", 1: "Positive"}


def parse_fasttext_line(line: str):
    """
    Parse one line from a fastText file.

    Returns (label_int, review_text) or None if the line is malformed.
    """
    line = line.strip()
    if not line:
        return None  # skip blank lines

    # The label is the first whitespace-delimited token
    parts = line.split(" ", 1)
    if len(parts) != 2:
        return None  # no space found -> malformed

    label_str, text = parts[0], parts[1].strip()

    if label_str not in LABEL_MAP:
        return None  # unknown label

    if not text:
        return None  # label present but no text

    return LABEL_MAP[label_str], text


def load_fasttext_file(filepath: str, max_samples: int = None):
    """
    Load a fastText-format file and return parallel lists of labels and texts.

    Parameters
    
    filepath    : path to the .txt file
    max_samples : if given, stop after reading this many VALID records.
                  Uses reservoir-style head sampling (first N valid records).
                  Pass None to load everything.

    Returns
    
    labels : list[int]   - 0 = Negative, 1 = Positive
    texts  : list[str]   - raw review text (before preprocessing)
    stats  : dict        - loading statistics
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    labels = []
    texts  = []
    total_lines   = 0
    malformed     = 0
    empty_text    = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1
            result = parse_fasttext_line(line)

            if result is None:
                # Distinguish between truly blank and malformed
                stripped = line.strip()
                if not stripped:
                    pass  # blank line — not counted as malformed
                else:
                    # Check if it has a label but empty text
                    parts = stripped.split(" ", 1)
                    if len(parts) == 1 or (len(parts) == 2 and not parts[1].strip()):
                        empty_text += 1
                    else:
                        malformed += 1
                continue

            labels.append(result[0])
            texts.append(result[1])

            if max_samples is not None and len(labels) >= max_samples:
                break

    stats = {
        "total_lines":   total_lines,
        "valid_records": len(labels),
        "malformed":     malformed,
        "empty_text":    empty_text,
        "positive":      labels.count(1),
        "negative":      labels.count(0),
    }

    return labels, texts, stats


def print_load_summary(split_name: str, stats: dict):
    """Print a readable summary of load statistics."""
    print(f"\n{'='*50}")
    print(f"  {split_name} Dataset Summary")
    print(f"{'='*50}")
    print(f"  Total lines read : {stats['total_lines']:,}")
    print(f"  Valid records    : {stats['valid_records']:,}")
    print(f"  Malformed lines  : {stats['malformed']:,}")
    print(f"  Empty-text lines : {stats['empty_text']:,}")
    print(f"  Positive reviews : {stats['positive']:,}")
    print(f"  Negative reviews : {stats['negative']:,}")
    print(f"{'='*50}\n")
