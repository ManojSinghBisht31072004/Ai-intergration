"""
RAG Response Schema Definition
Phase 2 – RAG Engineering
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChunkSource(BaseModel):
    chunkId: str = Field(..., description="Unique identifier of the retrieved chunk")
    snippet: str = Field(..., min_length=1, description="Relevant text excerpt from the chunk")

    @field_validator("chunkId")
    @classmethod
    def chunk_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunkId must not be empty or whitespace")
        return v.strip()

    @field_validator("snippet")
    @classmethod
    def snippet_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("snippet must not be empty or whitespace")
        return v.strip()


class RAGResponse(BaseModel):
    answer: str = Field(..., min_length=1, description="The answer to the user query")
    confidence: ConfidenceLevel = Field(..., description="Confidence level: high, medium, or low")
    sources: list[ChunkSource] = Field(
        ...,
        min_length=1,
        description="List of source chunks used to generate the answer"
    )

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer must not be empty or whitespace")
        return v.strip()

    @field_validator("sources")
    @classmethod
    def sources_not_empty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("sources must contain at least one chunk reference")
        return v


# JSON schema for use in LLM system prompts
RAG_JSON_SCHEMA = {
    "type": "object",
    "required": ["answer", "confidence", "sources"],
    "properties": {
        "answer": {
            "type": "string",
            "description": "Direct answer to the user query based on retrieved context"
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence level based on relevance and completeness of retrieved chunks"
        },
        "sources": {
            "type": "array",
            "minItems": 1,
            "description": "Source chunks used to generate the answer",
            "items": {
                "type": "object",
                "required": ["chunkId", "snippet"],
                "properties": {
                    "chunkId": {
                        "type": "string",
                        "description": "Unique identifier of the retrieved chunk"
                    },
                    "snippet": {
                        "type": "string",
                        "description": "Relevant excerpt from the chunk"
                    }
                }
            }
        }
    },
    "additionalProperties": False
}