from pydantic import BaseModel, Field
from typing import Literal, Optional

# ─── Request ───────────────────────────────────────────────
class EmailRequest(BaseModel):
    email: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The raw email text to analyze (max 8000 chars)"
    )

# ─── Debug / Cost Info ─────────────────────────────────────
class DebugInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float

# ─── Response ──────────────────────────────────────────────
class EmailResponse(BaseModel):
    tone: Literal["formal", "neutral", "urgent", "casual"]
    summary: str = Field(..., description="One-sentence summary of the email")
    suggestedReply: str = Field(..., description="A ready-to-send reply")
    debug: Optional[DebugInfo] = None

# ─── Error Response ────────────────────────────────────────
class ErrorResponse(BaseModel):
    error: str
    detail: str