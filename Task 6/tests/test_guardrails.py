import random
import string
import time
import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)

# ── Helpers ───────────────────────────────────────────────────────────────────

def random_query(length: int = 50) -> str:
    return "".join(random.choices(string.ascii_letters + " ", k=length))

def oversized_query() -> str:
    return "x" * 8001

VALID_JSON_RESPONSE = '{"answer": "Test answer", "sources": ["doc1.pdf"], "confidence": 0.9}'
INVALID_JSON_RESPONSE = "This is not JSON at all {broken"
FALLBACK = {
    "answer": "Unable to generate a valid response at this time. Please try again.",
    "sources": [],
    "confidence": 0.0
}

# ── Shared counters ───────────────────────────────────────────────────────────

results = {
    "valid_200": 0,
    "rejected_400": 0,
    "fallback": 0,
    "retried": 0,
    "timed_out_504": 0,
    "service_unavailable_503": 0,
    "crashed": 0,
}

# ── Individual test functions ─────────────────────────────────────────────────

def run_valid_request(i: int):
    # Patch where call_gemini is USED (in main.py), not where it's defined
    with patch("main.call_gemini", return_value=VALID_JSON_RESPONSE):
        resp = client.post("/rag/query", json={"query": random_query(100)})
        assert resp.status_code == 200, f"[{i}] Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "answer" in data
        assert "confidence" in data
        results["valid_200"] += 1


def run_oversized_request(i: int):
    resp = client.post("/rag/query", json={"query": oversized_query()})
    assert resp.status_code == 400, f"[{i}] Expected 400, got {resp.status_code}"
    assert "error" in resp.json()
    results["rejected_400"] += 1


def run_invalid_json_request(i: int):
    """LLM always returns bad JSON → validation fails → retry fails → safe fallback."""
    with patch("main.call_gemini", return_value=INVALID_JSON_RESPONSE):
        resp = client.post("/rag/query", json={"query": random_query(80)})
        assert resp.status_code == 200, f"[{i}] Expected 200 fallback, got {resp.status_code}"
        assert resp.json() == FALLBACK, f"[{i}] Expected fallback, got {resp.json()}"
        results["fallback"] += 1


def run_retry_on_rate_limit(i: int):
    """First call raises ResourceExhausted (429-equivalent), second succeeds."""
    call_count = {"n": 0}

    def flaky_gemini(prompt):
        call_count["n"] += 1
        if call_count["n"] == 1:
            from google.api_core.exceptions import ResourceExhausted
            raise ResourceExhausted("rate limited")
        return VALID_JSON_RESPONSE

    with patch("main.call_gemini", side_effect=flaky_gemini):
        resp = client.post("/rag/query", json={"query": random_query(60)})
        assert resp.status_code in (200, 503), f"[{i}] Got {resp.status_code}"
        results["retried"] += 1


def run_timeout_request(i: int):
    """call_gemini raises GatewayTimeoutError → endpoint returns 504."""
    from guardrails.exceptions import GatewayTimeoutError

    with patch("main.call_gemini", side_effect=GatewayTimeoutError("Timed out")):
        resp = client.post("/rag/query", json={"query": random_query(60)})
        assert resp.status_code == 504, f"[{i}] Expected 504, got {resp.status_code}"
        results["timed_out_504"] += 1


def run_server_error_exhausted(i: int):
    """All retries fail with ServiceUnavailable → 503."""
    from google.api_core.exceptions import ServiceUnavailable

    with patch("main.call_gemini", side_effect=ServiceUnavailable("down")):
        resp = client.post("/rag/query", json={"query": random_query(60)})
        assert resp.status_code in (503, 200), f"[{i}] Got {resp.status_code}"
        results["service_unavailable_503"] += 1


# ── 100-request runner ────────────────────────────────────────────────────────

def test_100_requests():
    logger.info("Starting 100-request guardrail stress test...")
    start = time.time()

    schedule = (
        [(run_valid_request, i) for i in range(50)] +
        [(run_oversized_request, i) for i in range(15)] +
        [(run_invalid_json_request, i) for i in range(15)] +
        [(run_retry_on_rate_limit, i) for i in range(10)] +
        [(run_timeout_request, i) for i in range(5)] +
        [(run_server_error_exhausted, i) for i in range(5)]
    )

    random.shuffle(schedule)

    for fn, i in schedule:
        try:
            fn(i)
        except AssertionError as e:
            results["crashed"] += 1
            logger.error(f"ASSERTION FAILED [{fn.__name__}][{i}]: {e}")
        except Exception as e:
            results["crashed"] += 1
            logger.error(f"UNEXPECTED CRASH [{fn.__name__}][{i}]: {type(e).__name__}: {e}")

    elapsed = time.time() - start

    print("\n" + "=" * 55)
    print("  GUARDRAIL TEST SUMMARY — 100 REQUESTS")
    print("=" * 55)
    print(f"  Valid 200 responses       : {results['valid_200']}")
    print(f"  Rejected 400 (too long)   : {results['rejected_400']}")
    print(f"  Fallback (invalid output) : {results['fallback']}")
    print(f"  Retried (429/5xx)         : {results['retried']}")
    print(f"  Timed out 504             : {results['timed_out_504']}")
    print(f"  Service unavailable 503   : {results['service_unavailable_503']}")
    print(f"  CRASHED (unexpected)      : {results['crashed']}")
    print(f"  Total time                : {elapsed:.2f}s")
    print("=" * 55)

    assert results["crashed"] == 0, f"Test suite had {results['crashed']} unexpected crashes!"