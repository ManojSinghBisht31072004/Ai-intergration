import logging

import google.generativeai as genai

from guardrails.config import GEMINI_API_KEY, GEMINI_MODEL, REQUEST_TIMEOUT_SECONDS
from guardrails.exceptions import GatewayTimeoutError

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config={
        "temperature": 0.2,
        "response_mime_type": "application/json",
    }
)


def build_prompt(user_query: str) -> str:
    return f"""You are a helpful assistant in a RAG system.
Answer the user's question using the provided context.

Respond ONLY with a valid JSON object — no markdown, no extra text — in this exact format:
{{
  "answer": "<your answer here>",
  "sources": ["<source1>", "<source2>"],
  "confidence": <float between 0.0 and 1.0>
}}

User question: {user_query}
"""


def call_gemini(prompt: str) -> str:
    """
    Synchronous Gemini call with SDK-level timeout.
    Raises GatewayTimeoutError on timeout, lets SDK exceptions bubble for retry_handler.
    """
    logger.debug(f"Calling Gemini model: {GEMINI_MODEL}")
    try:
        response = _model.generate_content(
            prompt,
            request_options={"timeout": REQUEST_TIMEOUT_SECONDS}
        )
        return response.text
    except Exception as e:
        if "timeout" in str(e).lower() or "deadline" in str(e).lower():
            raise GatewayTimeoutError(f"Gemini timed out after {REQUEST_TIMEOUT_SECONDS}s")
        raise