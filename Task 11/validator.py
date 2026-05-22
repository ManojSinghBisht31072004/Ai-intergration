"""
RAG Response Validator
Phase 2 – RAG Engineering
"""

import json
import logging
from dataclasses import dataclass, field
from pydantic import ValidationError
from schema import RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    response: RAGResponse | None = None
    errors: list[str] = field(default_factory=list)
    raw_json: dict | None = None


def validate_rag_response(raw: str | dict) -> ValidationResult:
    """
    Validate a RAG response against the RAGResponse schema.

    Args:
        raw: Either a JSON string or already-parsed dict from the LLM.

    Returns:
        ValidationResult with is_valid flag, parsed response (if valid), and errors.
    """
    # Step 1: Parse JSON if string
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON: {str(e)}"]
            )
    else:
        data = raw

    # Step 2: Validate against Pydantic schema
    try:
        response = RAGResponse(**data)
        return ValidationResult(
            is_valid=True,
            response=response,
            raw_json=data
        )
    except ValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        logger.warning(f"Schema validation failed: {errors}")
        return ValidationResult(
            is_valid=False,
            errors=errors,
            raw_json=data
        )
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        return ValidationResult(
            is_valid=False,
            errors=[f"Unexpected error: {str(e)}"],
            raw_json=data if isinstance(data, dict) else None
        )


def extract_json_from_text(text: str) -> str | None:
    """
    Extract JSON object from LLM text that may have surrounding prose.
    Tries to find a {...} block in the response.
    """
    text = text.strip()

    # Try direct parse first
    if text.startswith("{"):
        return text

    # Look for JSON block in markdown code fences
    import re
    fence_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    # Look for raw JSON object
    brace_pattern = r"(\{.*\})"
    match = re.search(brace_pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    return None