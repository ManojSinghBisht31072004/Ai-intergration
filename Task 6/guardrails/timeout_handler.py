import asyncio
import logging
from typing import Callable, Awaitable, TypeVar

from guardrails.config import REQUEST_TIMEOUT_SECONDS
from guardrails.exceptions import GatewayTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def call_with_timeout(
    coro: Awaitable[T],
    timeout_s: int = REQUEST_TIMEOUT_SECONDS
) -> T:
    """
    Awaits coro with a timeout. Raises GatewayTimeoutError on timeout.
    Caller should catch this and return HTTP 504.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out after {timeout_s} seconds.")
        raise GatewayTimeoutError(
            f"LLM did not respond within {timeout_s} seconds."
        )