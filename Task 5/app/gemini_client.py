"""
gemini_client.py
----------------
5-key rotation manager for Gemini API.
If a key fails (quota, rate limit, auth error), it seamlessly
rotates to the next key and retries THE SAME call from where it left off.
"""

import os
import time
import logging
from typing import Any, Callable
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── load all 5 keys ──────────────────────────────────────────────────────────
GEMINI_KEYS: list[str] = [
    k for k in [
        os.getenv("GEMINI_KEY_1"),
        os.getenv("GEMINI_KEY_2"),
        os.getenv("GEMINI_KEY_3"),
        os.getenv("GEMINI_KEY_4"),
        os.getenv("GEMINI_KEY_5"),
    ]
    if k  # skip None / empty strings
]

if not GEMINI_KEYS:
    raise EnvironmentError("No Gemini API keys found. Check your .env file.")

# ── errors that should trigger a key rotation ────────────────────────────────
ROTATE_ON = (
    "quota",
    "rate",
    "resource exhausted",
    "api key",
    "invalid api key",
    "permission denied",
    "429",
    "403",
)


class GeminiKeyManager:
    """
    Wraps every Gemini call. On failure, rotates to the next key
    and retries the exact same call — transparently.
    """

    def __init__(self):
        self.keys = GEMINI_KEYS
        self.current_index = 0
        self.key_errors: dict[int, list[str]] = {i: [] for i in range(len(self.keys))}
        logger.info(f"GeminiKeyManager ready with {len(self.keys)} key(s).")

    @property
    def current_key(self) -> str:
        return self.keys[self.current_index]

    def _should_rotate(self, error: Exception) -> bool:
        msg = str(error).lower()
        return any(trigger in msg for trigger in ROTATE_ON)

    def _rotate(self) -> bool:
        """Move to the next key. Returns False if all keys exhausted."""
        next_index = (self.current_index + 1) % len(self.keys)
        if next_index == 0 and self.current_index == len(self.keys) - 1:
            return False  # wrapped all the way around
        self.current_index = next_index
        logger.warning(f"Rotated to Gemini key #{self.current_index + 1}")
        return True

    def call_with_rotation(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Execute fn(*args, **kwargs) using the current key.
        On a rotatable error, switch key and retry.
        fn must accept `api_key` as its first positional argument
        OR we set genai.configure() before calling it.
        """
        tried = set()

        while len(tried) < len(self.keys):
            if self.current_index in tried:
                self._rotate()
                continue

            tried.add(self.current_index)
            key = self.current_key

            try:
                genai.configure(api_key=key)
                result = fn(*args, **kwargs)
                return result

            except Exception as e:
                error_msg = str(e)
                self.key_errors[self.current_index].append(error_msg)
                logger.error(
                    f"Key #{self.current_index + 1} failed: {error_msg[:120]}"
                )

                if self._should_rotate(e):
                    rotated = self._rotate()
                    if not rotated:
                        break
                    time.sleep(0.5)  # brief pause before retry
                else:
                    # non-quota error (bad request, etc.) — don't rotate, raise
                    raise

        raise RuntimeError(
            f"All {len(self.keys)} Gemini API keys failed. "
            "Check quotas or network connectivity."
        )

    def status(self) -> dict:
        """Return key health status — useful for /health endpoint."""
        return {
            "total_keys": len(self.keys),
            "active_key_index": self.current_index + 1,
            "errors_per_key": {
                f"key_{i + 1}": len(errs)
                for i, errs in self.key_errors.items()
            },
        }


# ── singleton ─────────────────────────────────────────────────────────────────
key_manager = GeminiKeyManager()


# ── convenience wrappers ──────────────────────────────────────────────────────

# def embed_text(text: str) -> list[float]:
#     """Embed a single string. Rotates keys on failure."""

#     def _embed(t: str) -> list[float]:
#         result = genai.embed_content(
#             model="models/text-embedding-004",
#             content=t,
#             task_type="retrieval_document",
#         )
#         return result["embedding"]

#     return key_manager.call_with_rotation(_embed, text)


# def embed_query(text: str) -> list[float]:
#     """Embed a query string (different task_type for better retrieval)."""

#     def _embed(t: str) -> list[float]:
#         result = genai.embed_content(
#             model="models/text-embedding-004",
#             content=t,
#             task_type="retrieval_query",
#         )
#         return result["embedding"]

#     return key_manager.call_with_rotation(_embed, text) 
# 2 try TO DO SOME CHANGES
def embed_text(text: str) -> list[float]:
    def _embed(t: str) -> list[float]:
        result = genai.embed_content(
            model="models/gemini-embedding-001",  # fixed
            content=t,
            task_type="retrieval_document",
        )
        return result["embedding"]
    return key_manager.call_with_rotation(_embed, text)


def embed_query(text: str) -> list[float]:
    def _embed(t: str) -> list[float]:
        result = genai.embed_content(
            model="models/gemini-embedding-001",  # fixed
            content=t,
            task_type="retrieval_query",
        )
        return result["embedding"]
    return key_manager.call_with_rotation(_embed, text)

def generate_answer(prompt: str, max_tokens: int = 1024) -> str:
    def _generate(p: str, mt: int) -> str:
        model = genai.GenerativeModel("models/gemini-3.1-flash-lite")  # fixed
        response = model.generate_content(
            p,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=mt,
                temperature=0.1,
            ),
        )
        return response.text
    return key_manager.call_with_rotation(_generate, prompt, max_tokens)