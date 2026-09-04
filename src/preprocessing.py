"""
preprocessing.py
----------------
Text cleaning for product reviews.

Design decisions:
  - Lowercase everything (case doesn't carry sentiment).
  - Remove HTML tags (some reviews contain markup artefacts).
  - Collapse multiple spaces/newlines.
  - Keep negations (not, never, no) — they carry strong sentiment signal.
  - Keep punctuation removal minimal; exclamation marks etc. can carry
    sentiment but TF-IDF does not benefit from them, so we strip non-alpha
    characters EXCEPT apostrophes (don't, isn't, won't).
  - Strip leading/trailing whitespace.

The same function is used at training time and at prediction time so that
the model always sees text in the same form.
"""

import re


# Matches any HTML tag like <br />, <b>, </p> etc.
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Matches characters that are NOT letters, digits, spaces, or apostrophes.
NON_ALPHA_PATTERN = re.compile(r"[^a-z0-9\s']")

# Collapses two or more whitespace characters (including \n, \t) into one space.
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_review(text: str) -> str:
    """
    Clean a single review string.

    Steps:
    1. Lowercase
    2. Remove HTML tags
    3. Remove non-alphanumeric characters (keep apostrophes for contractions)
    4. Collapse extra whitespace
    5. Strip edges

    Parameters
    ----------
    text : raw review string

    Returns
    -------
    Cleaned string, or empty string if input is None/empty.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Strip HTML tags (e.g. <br />, <b>bold</b>)
    text = HTML_TAG_PATTERN.sub(" ", text)

    # 3. Remove characters that are not letters, digits, spaces, or apostrophes
    #    Apostrophes preserved so "don't", "won't", "isn't" stay intact.
    text = NON_ALPHA_PATTERN.sub(" ", text)

    # 4. Collapse multiple spaces / newlines into a single space
    text = WHITESPACE_PATTERN.sub(" ", text)

    # 5. Strip leading/trailing whitespace
    return text.strip()


def preprocess_texts(texts: list) -> list:
    """
    Apply clean_review to a list of review strings.

    Parameters
    ----------
    texts : list of raw review strings

    Returns
    -------
    list of cleaned review strings
    """
    return [clean_review(t) for t in texts]
