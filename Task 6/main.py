import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from guardrails import (
    check_input_length,
    parse_and_validate,
    call_with_retry,
    InputTooLongError,
    GatewayTimeoutError,
    RetryExhaustedError,
)
from guardrails.config import SAFE_FALLBACK_RESPONSE, REQUEST_TIMEOUT_SECONDS, MAX_RETRIES
from guardrails.llm_client import build_prompt, call_gemini

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Guardrail RAG API starting up.")
    yield
    logger.info("Guardrail RAG API shutting down.")


app = FastAPI(title="RAG Guardrail API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


def _guarded_llm_call(prompt: str) -> str:
    """
    Synchronous LLM call wrapped with retry.
    Timeout is enforced inside call_gemini via the SDK timeout param.
    """
    return call_with_retry(
        fn=lambda: call_gemini(prompt),
        max_retries=MAX_RETRIES
    )


@app.post("/rag/query")
def rag_query(request: QueryRequest):
    # ① Input length guard
    try:
        check_input_length(request.query)
    except InputTooLongError as e:
        logger.warning(f"Input rejected: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

    prompt = build_prompt(request.query)

    # ③ + ④ Retry + Timeout
    raw_text = None
    try:
        raw_text = _guarded_llm_call(prompt)
    except GatewayTimeoutError as e:
        logger.error(f"Timeout: {e}")
        return JSONResponse(status_code=504, content={"error": str(e)})
    except RetryExhaustedError as e:
        logger.error(f"Retry exhausted: {e}")
        return JSONResponse(status_code=503, content={"error": "Service temporarily unavailable."})
    except Exception as e:
        logger.error(f"Non-retryable LLM error: {e}")
        return JSONResponse(status_code=502, content={"error": "LLM call failed."})

    # ② Output schema validation — retry once, then safe fallback
    result = parse_and_validate(raw_text)

    if result is None:
        logger.warning("Output validation failed on first attempt. Retrying once.")
        try:
            raw_text_retry = call_gemini(prompt)
            result = parse_and_validate(raw_text_retry)
        except Exception as e:
            logger.error(f"Retry for schema fix failed: {e}")
            result = None

        if result is None:
            logger.warning("Using safe fallback response.")
            return JSONResponse(status_code=200, content=SAFE_FALLBACK_RESPONSE)

    return JSONResponse(status_code=200, content=result.model_dump())


@app.get("/health")
def health():
    return {"status": "ok"}