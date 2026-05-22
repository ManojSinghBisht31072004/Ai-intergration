"""
RAG Engine – Phase 2
Groq API + structured output + retry/fallback
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
from schema import RAGResponse, RAG_JSON_SCHEMA, ConfidenceLevel, ChunkSource
from validator import validate_rag_response, extract_json_from_text, ValidationResult

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL          = "llama-3.3-70b-versatile"   # change to any Groq-hosted model you prefer
MAX_RETRIES    = 2                            # number of retry attempts after first failure
RETRY_DELAY    = 1.0                          # seconds between retries
TEMPERATURE    = 0.1                          # low temp for deterministic structured output


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise RAG (Retrieval-Augmented Generation) assistant.

Your task: answer the user's question using ONLY the retrieved context chunks provided.

You MUST respond with a single valid JSON object matching this exact schema — no prose, no markdown fences, no extra keys:

{schema}

Rules:
1. "answer": concise, factual answer derived from the context. If context is insufficient, say so.
2. "confidence":
   - "high"   → context directly and fully answers the question
   - "medium" → context partially answers or requires inference
   - "low"    → context is tangential or insufficient
3. "sources": list every chunk you used. Include its chunkId and a short snippet (≤ 150 chars) that supports your answer.
4. Do NOT invent information not present in the context.
5. Output ONLY the JSON object. Nothing else.
""".format(schema=json.dumps(RAG_JSON_SCHEMA, indent=2))


# ── Chunk / retrieval types ───────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float = 0.0          # similarity score from vector store


@dataclass
class RAGQueryResult:
    query: str
    response: RAGResponse | None
    is_valid: bool
    attempts: int
    errors: list[str] = field(default_factory=list)
    used_fallback: bool = False
    latency_ms: float = 0.0


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, question: str, chunks: list[RetrievedChunk]) -> RAGQueryResult:
        """
        Run a RAG query with structured output, retry, and fallback.

        Args:
            question: User's natural-language question.
            chunks:   Retrieved context chunks from the vector store.

        Returns:
            RAGQueryResult with validated RAGResponse (or fallback).
        """
        start = time.perf_counter()
        context_block = self._format_context(chunks)
        user_message  = self._format_user_message(question, context_block)

        all_errors: list[str] = []
        attempts = 0

        # ── Attempt with retries ──────────────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 2):   # +2 = initial + retries
            attempts = attempt
            logger.info(f"Query attempt {attempt}/{MAX_RETRIES + 1}")

            raw_text = self._call_llm(user_message, attempt)
            if raw_text is None:
                all_errors.append(f"Attempt {attempt}: LLM call failed")
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue

            # Try to extract JSON if LLM wrapped it in prose/fences
            json_str = extract_json_from_text(raw_text)
            if json_str is None:
                all_errors.append(f"Attempt {attempt}: No JSON found in response")
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue

            result: ValidationResult = validate_rag_response(json_str)
            if result.is_valid:
                latency = (time.perf_counter() - start) * 1000
                logger.info(f"Valid response on attempt {attempt} ({latency:.0f}ms)")
                return RAGQueryResult(
                    query=question,
                    response=result.response,
                    is_valid=True,
                    attempts=attempts,
                    errors=all_errors,
                    latency_ms=latency
                )
            else:
                all_errors.extend([f"Attempt {attempt} – {e}" for e in result.errors])
                logger.warning(f"Attempt {attempt} invalid: {result.errors}")
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        # ── All attempts failed → fallback ────────────────────────────────────
        logger.error(f"All {MAX_RETRIES + 1} attempts failed. Using fallback.")
        fallback = self._build_fallback(question, chunks, all_errors)
        latency  = (time.perf_counter() - start) * 1000
        return RAGQueryResult(
            query=question,
            response=fallback,
            is_valid=False,             # mark False so metrics count it correctly
            attempts=attempts,
            errors=all_errors,
            used_fallback=True,
            latency_ms=latency
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_llm(self, user_message: str, attempt: int) -> str | None:
        """Call the Groq chat completion API."""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message}
                ],
                response_format={"type": "json_object"},   # Groq JSON mode
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call error (attempt {attempt}): {e}")
            return None

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        lines = []
        for c in chunks:
            lines.append(f"[{c.chunk_id}] (score={c.score:.3f})\n{c.text}")
        return "\n\n---\n\n".join(lines)

    @staticmethod
    def _format_user_message(question: str, context: str) -> str:
        return (
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            "Respond with the JSON object only."
        )

    @staticmethod
    def _build_fallback(
        question: str,
        chunks: list[RetrievedChunk],
        errors: list[str]
    ) -> RAGResponse:
        """
        Construct a minimal valid RAGResponse when all LLM attempts fail.
        Uses the top chunk text as the answer to ensure schema compliance.
        """
        if chunks:
            top = chunks[0]
            answer  = f"[Fallback] Based on retrieved context: {top.text[:300]}"
            sources = [ChunkSource(chunkId=top.chunk_id, snippet=top.text[:150])]
        else:
            answer  = "[Fallback] No context available to answer this question."
            sources = [ChunkSource(chunkId="fallback-chunk-000", snippet="No context retrieved.")]

        return RAGResponse(
            answer=answer,
            confidence=ConfidenceLevel.LOW,
            sources=sources
        )