"""
integration_test.py
-------------------
Full end-to-end integration test.
Run from the project root:
    python integration_test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from fastapi.testclient import TestClient
from main import app
from predict import predict_sentiment

client = TestClient(app)
all_ok = True

def check(name, got, expected):
    global all_ok
    ok = (got == expected)
    if not ok:
        all_ok = False
    tag = "PASS" if ok else "FAIL"
    return tag, ok


# ── Backend tests ──────────────────────────────────────────────────────────
print("\nBackend endpoint tests:")
tests = [
    ("GET /health",        "GET",  "/health",  None,               200),
    ("POST positive",      "POST", "/predict", {"review": "Amazing product! Works perfectly and arrived fast."}, 200),
    ("POST negative",      "POST", "/predict", {"review": "Terrible, broke in 2 days. Total waste."}, 200),
    ("POST negation",      "POST", "/predict", {"review": "Not good at all. I will never buy again."}, 200),
    ("POST empty",         "POST", "/predict", {"review": ""},      422),
    ("POST whitespace",    "POST", "/predict", {"review": "   "},   422),
    ("POST too short",     "POST", "/predict", {"review": "ok"},    422),
    ("POST too long",      "POST", "/predict", {"review": "A" * 5001}, 422),
    ("POST missing field", "POST", "/predict", {"text": "something"}, 422),
]

for name, method, path, body, expected_status in tests:
    r = client.get(path) if method == "GET" else client.post(path, json=body)
    tag, _ = check(name, r.status_code, expected_status)
    extra = ""
    if r.status_code == 200:
        extra = "  sentiment=" + r.json().get("sentiment", "")
    print(f"  [{tag}] {name:<28} status={r.status_code}{extra}")


# ── Predict module tests ───────────────────────────────────────────────────
print("\nPredict module tests:")
samples = [
    ("Absolutely love this product!",                 "Positive"),
    ("Waste of money, quality is awful.",              "Negative"),
    ("Not good at all, never again.",                  "Negative"),
    ("Works exactly as described. Very happy with it.", "Positive"),
    ("Do not buy this. Very disappointing purchase.",   "Negative"),
    ("Excellent value, highly recommended.",            "Positive"),
    ("Stopped working after one week. Terrible.",       "Negative"),
]

for text, expected in samples:
    result = predict_sentiment(text)
    got = result["label"]
    tag, _ = check(text, got, expected)
    print(f"  [{tag}] expected={expected:<8}  got={got:<8}  | {text[:55]}")

print()
if all_ok:
    print("ALL INTEGRATION TESTS PASSED")
else:
    print("SOME TESTS FAILED — see above")
    sys.exit(1)
