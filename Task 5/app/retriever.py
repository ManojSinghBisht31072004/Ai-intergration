"""
retriever.py
------------
Query pipeline:
  1. Embed the question
  2. Retrieve top-k chunks from ChromaDB
  3. Build a grounded prompt
  4. Call Gemini LLM (with key rotation)
  5. Return { answer, sources, grounded }

Strict grounding: model is told to say "I don't know" if context lacks the answer.
Source validation: cited chunk IDs are checked against the retrieved set.
"""

import json
import logging
import re

try:
    from app.gemini_client import embed_query, generate_answer
    from app.vectorstore import retrieve_top_k, collection_count
except ModuleNotFoundError:
    from gemini_client import embed_query, generate_answer
    from vectorstore import retrieve_top_k, collection_count

logger = logging.getLogger(__name__)

TOP_K = 5

SYSTEM_PROMPT = """You are a precise knowledge base assistant.

RULES (follow strictly):
1. Answer ONLY using the context chunks provided below.
2. If the answer is not present in the context, respond with exactly:
   "I don't know based on the provided documents."
3. Do NOT use any external knowledge or make assumptions.
4. After your answer, list the chunk IDs you used as sources.

OUTPUT FORMAT (valid JSON only, no markdown fences):
{{
  "answer": "<your answer or the I don't know phrase>",
  "sources": [
    {{"chunk_id": "<id>", "snippet": "<15-word excerpt from that chunk>"}}
  ]
}}

CONTEXT CHUNKS:
{context}

QUESTION: {question}
"""


def _build_context_block(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c['chunk_id']}] {c['text']}")
    return "\n\n".join(lines)


def _parse_response(raw: str) -> dict:
    """Parse LLM JSON output. Fallback to plain text if JSON is malformed."""
    cleaned = raw.strip()
    # strip accidental markdown fences
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # graceful degradation: treat whole response as answer, no sources
        return {"answer": cleaned, "sources": []}


def _validate_sources(sources: list[dict], retrieved_chunk_ids: set[str]) -> tuple[list[dict], bool]:
    """
    Check that every cited chunk_id was actually in the retrieved set.
    Returns (validated_sources, hallucination_detected).
    """
    valid = []
    hallucinated = False

    for src in sources:
        cid = src.get("chunk_id", "")
        if cid in retrieved_chunk_ids:
            valid.append(src)
        else:
            logger.warning(f"Hallucinated chunk_id cited: '{cid}' not in retrieved set")
            hallucinated = True

    return valid, hallucinated


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG query pipeline.
    Returns:
      {
        "answer": str,
        "sources": [{"chunk_id": str, "snippet": str}],
        "grounded": bool,          # False if model said "I don't know"
        "hallucination_detected": bool,  # True if model cited invalid chunk IDs
        "retrieved_chunks": int,
      }
    """
    if collection_count() == 0:
        return {
            "answer": "No documents have been uploaded yet. Please upload a document first.",
            "sources": [],
            "grounded": False,
            "hallucination_detected": False,
            "retrieved_chunks": 0,
        }

    # 1. Embed question
    q_embedding = embed_query(question)

    # 2. Retrieve top-k chunks
    chunks = retrieve_top_k(q_embedding, k=top_k)
    retrieved_ids = {c["chunk_id"] for c in chunks}

    if not chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "grounded": False,
            "hallucination_detected": False,
            "retrieved_chunks": 0,
        }

    # 3. Build grounded prompt
    context_block = _build_context_block(chunks)
    prompt = SYSTEM_PROMPT.format(context=context_block, question=question)

    # 4. Call Gemini (key rotation is automatic)
    raw_response = generate_answer(prompt, max_tokens=1024)
    logger.debug(f"Raw LLM response: {raw_response[:300]}")

    # 5. Parse + validate
    parsed = _parse_response(raw_response)
    answer = parsed.get("answer", "").strip()
    raw_sources = parsed.get("sources", [])

    validated_sources, hallucination = _validate_sources(raw_sources, retrieved_ids)

    grounded = "i don't know" not in answer.lower()

    return {
        "answer": answer,
        "sources": validated_sources,
        "grounded": grounded,
        "hallucination_detected": hallucination,
        "retrieved_chunks": len(chunks),
    }