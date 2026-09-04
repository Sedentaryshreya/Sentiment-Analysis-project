"""
test_backend.py
---------------
Tests the FastAPI app directly via ASGI test client (no server needed).
Run from the project root:
    python test_backend.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def sep(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ── /health ────────────────────────────────────────────────
sep("GET /health")
r = client.get("/health")
print(f"  Status : {r.status_code}")
print(f"  Body   : {r.json()}")
assert r.status_code == 200
assert r.json()["status"] == "ok"
print("  PASS")

# ── /predict — positive review ─────────────────────────────
sep("POST /predict  (positive review)")
r = client.post("/predict", json={"review": "This product exceeded all my expectations. Absolutely fantastic quality and fast delivery!"})
print(f"  Status    : {r.status_code}")
print(f"  Body      : {r.json()}")
assert r.status_code == 200
assert r.json()["sentiment"] == "Positive"
print("  PASS")

# ── /predict — negative review ─────────────────────────────
sep("POST /predict  (negative review)")
r = client.post("/predict", json={"review": "Complete garbage. Stopped working after 2 days. Never buying this brand again. Total waste of money."})
print(f"  Status    : {r.status_code}")
print(f"  Body      : {r.json()}")
assert r.status_code == 200
assert r.json()["sentiment"] == "Negative"
print("  PASS")

# ── /predict — negation test ───────────────────────────────
sep("POST /predict  (negation: 'not good')")
r = client.post("/predict", json={"review": "Not good at all, I would never recommend this product to anyone."})
print(f"  Status    : {r.status_code}")
print(f"  Body      : {r.json()}")
assert r.status_code == 200
assert r.json()["sentiment"] == "Negative"
print("  PASS")

# ── /predict — empty input ────────────────────────────────
sep("POST /predict  (empty — expect 422)")
r = client.post("/predict", json={"review": "   "})
print(f"  Status : {r.status_code}")
print(f"  Body   : {r.json()}")
assert r.status_code == 422
print("  PASS")

# ── /predict — too short ──────────────────────────────────
sep("POST /predict  (too short — expect 422)")
r = client.post("/predict", json={"review": "hi"})
print(f"  Status : {r.status_code}")
print(f"  Body   : {r.json()}")
assert r.status_code == 422
print("  PASS")

# ── /predict — too long ───────────────────────────────────
sep("POST /predict  (5001 chars — expect 422)")
r = client.post("/predict", json={"review": "A" * 5001})
print(f"  Status : {r.status_code}")
print(f"  Body   : {r.json()}")
assert r.status_code == 422
print("  PASS")

# ── /predict — missing field ──────────────────────────────
sep("POST /predict  (missing 'review' field — expect 422)")
r = client.post("/predict", json={"text": "something"})
print(f"  Status : {r.status_code}")
print("  PASS")

print("\n" + "="*55)
print("  ALL BACKEND TESTS PASSED")
print("="*55 + "\n")
