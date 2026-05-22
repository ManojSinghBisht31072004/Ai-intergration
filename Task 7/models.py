from pydantic import BaseModel, Field
from typing import Literal


class LeadRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Full name of the lead")
    company: str = Field(..., min_length=1, description="Company name of the lead")
    notes: str = Field(..., min_length=1, description="Raw notes about the lead")


class LeadResponse(BaseModel):
    summary: str = Field(..., description="2-3 sentence summary of the lead")
    suggestedFollowUp: str = Field(..., description="One concrete follow-up action")
    sentimentScore: Literal["positive", "neutral", "negative"] = Field(
        ..., description="Sentiment of the lead"
    )