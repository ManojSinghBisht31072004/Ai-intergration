import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

MAX_INPUT_CHARS = 8000
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

SAFE_FALLBACK_RESPONSE = {
    "answer": "Unable to generate a valid response at this time. Please try again.",
    "sources": [],
    "confidence": 0.0
}