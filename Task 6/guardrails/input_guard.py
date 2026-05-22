from guardrails.config import MAX_INPUT_CHARS
from guardrails.exceptions import InputTooLongError


def check_input_length(text: str) -> None:
    """
    Raises InputTooLongError if text exceeds MAX_INPUT_CHARS.
    Caller should catch this and return HTTP 400.
    """
    length = len(text)
    if length > MAX_INPUT_CHARS:
        raise InputTooLongError(
            f"Input length {length} exceeds maximum allowed {MAX_INPUT_CHARS} characters."
        )