import time
import random
import logging
from typing import Callable, Any

from guardrails.exceptions import RetryExhaustedError
from guardrails.config import MAX_RETRIES

logger = logging.getLogger(__name__)

# Status codes considered transient (safe to retry)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def is_retryable_exception(exc: Exception) -> bool:
    """
    Returns True if the exception is transient and worth retrying.
    Checks for google.api_core exceptions and HTTP status codes.
    """
    exc_type = type(exc).__name__
    exc_str = str(exc)

    # google-generativeai raises these for transient errors
    retryable_names = {
        "ResourceExhausted",   # 429 rate limit
        "ServiceUnavailable",  # 503
        "InternalServerError", # 500
        "DeadlineExceeded",    # timeout at SDK level
        "Aborted",
    }
    if exc_type in retryable_names:
        return True

    # Fallback: check if message contains a retryable status code
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in exc_str:
            return True

    return False


def call_with_retry(fn: Callable[[], Any], max_retries: int = MAX_RETRIES) -> Any:
    """
    Calls fn() up to max_retries times with exponential backoff + jitter.
    Only retries on transient errors. Raises RetryExhaustedError if all fail.
    Raises immediately on non-retryable (4xx) errors.
    """
    last_exc = None

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if not is_retryable_exception(exc):
                logger.warning(f"Non-retryable error on attempt {attempt + 1}: {exc}")
                raise  # 4xx and other permanent errors bubble up immediately

            last_exc = exc
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}/{max_retries}. "
                    f"Retrying in {wait:.2f}s. Error: {exc}"
                )
                time.sleep(wait)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {exc}")

    raise RetryExhaustedError(f"Failed after {max_retries} attempts. Last error: {last_exc}")