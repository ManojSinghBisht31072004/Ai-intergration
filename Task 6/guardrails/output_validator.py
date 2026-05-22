import json
from typing import Optional

from pydantic import BaseModel, ValidationError, Field

from guardrails.exceptions import OutputValidationError


class LLMResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


def parse_and_validate(raw_text: str) -> Optional[LLMResponse]:
    """
    Attempts to parse raw_text as JSON and validate against LLMResponse schema.
    Returns LLMResponse on success, None on any failure.
    Never raises — callers decide retry/fallback logic.
    """
    try:
        # Strip markdown code fences if model wraps output
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        data = json.loads(cleaned)
        return LLMResponse(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None