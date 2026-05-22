"""
Test Runner – 20 RAG Queries
Phase 2 – RAG Engineering

Usage:
    GROQ_API_KEY=<key> python test_runner.py

Outputs:
    - Console: per-query results + final metrics
    - test_results.json: machine-readable results
"""

import json
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from rag_engine import RAGEngine, RetrievedChunk, RAGQueryResult

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_runner")

# ── Sample corpus (replace with your actual vector store retrieval) ───────────
# Each tuple: (query, [RetrievedChunk, ...])
# In production: swap get_test_cases() with real retrieval logic.

def get_test_cases() -> list[tuple[str, list[RetrievedChunk]]]:
    """
    20 test queries with pre-populated mock chunks.
    Replace chunk text with actual retrieved chunks from your vector store.
    """
    return [
        (
            "What is retrieval-augmented generation?",
            [
                RetrievedChunk("chunk-001", "Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative model. The retrieval system fetches relevant documents, which are then passed to the LLM as context.", 0.95),
                RetrievedChunk("chunk-002", "RAG was introduced to reduce hallucinations by grounding LLM responses in factual, retrieved documents.", 0.88),
            ]
        ),
        (
            "How does vector similarity search work?",
            [
                RetrievedChunk("chunk-010", "Vector similarity search converts text into high-dimensional embeddings and measures cosine similarity between query and document vectors.", 0.93),
                RetrievedChunk("chunk-011", "FAISS and Pinecone are popular libraries for approximate nearest-neighbor search over embedding vectors.", 0.85),
            ]
        ),
        (
            "What is chunking in RAG pipelines?",
            [
                RetrievedChunk("chunk-020", "Chunking splits large documents into smaller segments so each fits within the LLM context window and retrieval is more precise.", 0.91),
                RetrievedChunk("chunk-021", "Common chunking strategies include fixed-size chunks, sentence-level splits, and recursive character splitting.", 0.87),
            ]
        ),
        (
            "What embedding models are commonly used?",
            [
                RetrievedChunk("chunk-030", "OpenAI's text-embedding-ada-002 and open-source models like BAAI/bge-base-en are widely used for document embeddings.", 0.90),
                RetrievedChunk("chunk-031", "Sentence-transformers provide lightweight embedding models suitable for local deployment.", 0.82),
            ]
        ),
        (
            "How do you evaluate RAG pipeline quality?",
            [
                RetrievedChunk("chunk-040", "RAG evaluation metrics include faithfulness (answer grounded in context), answer relevance, and context recall.", 0.92),
                RetrievedChunk("chunk-041", "RAGAS is a popular framework for automated RAG evaluation using LLM-as-a-judge scoring.", 0.88),
            ]
        ),
        (
            "What is the difference between sparse and dense retrieval?",
            [
                RetrievedChunk("chunk-050", "Sparse retrieval (BM25) uses term frequency statistics; dense retrieval uses neural embeddings to capture semantic meaning.", 0.94),
                RetrievedChunk("chunk-051", "Hybrid retrieval combines both sparse and dense methods to improve recall.", 0.86),
            ]
        ),
        (
            "What is re-ranking in retrieval systems?",
            [
                RetrievedChunk("chunk-060", "Re-ranking is a post-retrieval step where a cross-encoder model scores and reorders the initial retrieved candidates.", 0.89),
                RetrievedChunk("chunk-061", "Cohere Rerank and BGE rerankers are commonly used to improve precision after initial retrieval.", 0.83),
            ]
        ),
        (
            "How does context window size affect RAG?",
            [
                RetrievedChunk("chunk-070", "Larger context windows allow more chunks to be included, improving answer completeness but increasing latency and cost.", 0.90),
                RetrievedChunk("chunk-071", "Lost-in-the-middle research shows LLMs struggle to attend to information at the middle of long contexts.", 0.85),
            ]
        ),
        (
            "What is metadata filtering in RAG?",
            [
                RetrievedChunk("chunk-080", "Metadata filtering restricts retrieval to chunks matching attributes like date, author, or document type before vector search.", 0.91),
                RetrievedChunk("chunk-081", "Pre-filtering reduces the search space and improves retrieval precision for structured corpora.", 0.84),
            ]
        ),
        (
            "How do you handle multi-hop questions in RAG?",
            [
                RetrievedChunk("chunk-090", "Multi-hop questions require retrieving multiple documents and reasoning across them. Iterative retrieval or chain-of-thought prompting helps.", 0.88),
                RetrievedChunk("chunk-091", "HippoRAG and self-ask prompting are techniques for multi-step reasoning in RAG.", 0.80),
            ]
        ),
        (
            "What is a vector database?",
            [
                RetrievedChunk("chunk-100", "A vector database stores high-dimensional embedding vectors and supports efficient approximate nearest-neighbor queries.", 0.95),
                RetrievedChunk("chunk-101", "Examples include Pinecone, Weaviate, Qdrant, Chroma, and Milvus.", 0.90),
            ]
        ),
        (
            "How does RAG reduce hallucination?",
            [
                RetrievedChunk("chunk-110", "RAG reduces hallucination by providing the LLM with factual retrieved context, anchoring responses to source documents.", 0.94),
                RetrievedChunk("chunk-111", "Without retrieval, LLMs rely solely on parametric knowledge, which can be outdated or fabricated.", 0.87),
            ]
        ),
        (
            "What is the role of the system prompt in RAG?",
            [
                RetrievedChunk("chunk-120", "The system prompt instructs the LLM to answer only from retrieved context and defines output format requirements.", 0.92),
                RetrievedChunk("chunk-121", "Well-crafted system prompts significantly reduce hallucination and improve structured output compliance.", 0.88),
            ]
        ),
        (
            "What is document indexing?",
            [
                RetrievedChunk("chunk-130", "Document indexing converts raw documents into searchable embeddings stored in a vector database during the offline pipeline stage.", 0.91),
                RetrievedChunk("chunk-131", "Indexing typically involves loading, splitting, embedding, and upserting chunks into the vector store.", 0.86),
            ]
        ),
        (
            "How do you handle documents with no relevant answer?",
            [
                RetrievedChunk("chunk-140", "When retrieved chunks don't contain the answer, the model should indicate low confidence and state that the context is insufficient.", 0.89),
                RetrievedChunk("chunk-141", "Confidence thresholds can trigger a fallback response or request for document expansion.", 0.82),
            ]
        ),
        (
            "What is semantic chunking?",
            [
                RetrievedChunk("chunk-150", "Semantic chunking splits documents at natural semantic boundaries (topic shifts) rather than fixed character counts.", 0.90),
                RetrievedChunk("chunk-151", "This preserves coherent ideas within chunks, improving retrieval relevance.", 0.85),
            ]
        ),
        (
            "How is RAG different from fine-tuning?",
            [
                RetrievedChunk("chunk-160", "Fine-tuning bakes knowledge into model weights; RAG retrieves knowledge at inference time from an external store.", 0.93),
                RetrievedChunk("chunk-161", "RAG is easier to update (just re-index documents) while fine-tuning requires retraining.", 0.87),
            ]
        ),
        (
            "What is HyDE in RAG?",
            [
                RetrievedChunk("chunk-170", "HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to the query, embeds it, and uses that embedding for retrieval.", 0.91),
                RetrievedChunk("chunk-171", "HyDE improves retrieval when the query phrasing differs significantly from document phrasing.", 0.84),
            ]
        ),
        (
            "How do you measure retrieval recall?",
            [
                RetrievedChunk("chunk-180", "Retrieval recall measures what fraction of relevant documents were retrieved. Higher recall means fewer missed relevant chunks.", 0.92),
                RetrievedChunk("chunk-181", "It is computed as: relevant_retrieved / total_relevant, evaluated against a labeled ground-truth set.", 0.87),
            ]
        ),
        (
            "What is a knowledge graph RAG?",
            [
                RetrievedChunk("chunk-190", "Knowledge graph RAG augments retrieval with structured entity-relationship data, improving multi-hop reasoning.", 0.89),
                RetrievedChunk("chunk-191", "Microsoft GraphRAG is a notable example combining community detection with LLM-based summarization.", 0.83),
            ]
        ),
    ]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tests(api_key: str | None = None) -> None:
    engine = RAGEngine(api_key=api_key)
    test_cases = get_test_cases()
    assert len(test_cases) == 20, "Must have exactly 20 test cases"

    results: list[dict] = []
    valid_count   = 0
    fallback_count = 0
    total_attempts = 0

    print("\n" + "=" * 70)
    print("  RAG Phase 2 – Structured Output Test Run (20 queries)")
    print("=" * 70 + "\n")

    for i, (query, chunks) in enumerate(test_cases, start=1):
        print(f"[{i:02d}/20] {query[:65]}...")
        result: RAGQueryResult = engine.query(query, chunks)
        total_attempts += result.attempts

        status = "✓ VALID"
        if result.used_fallback:
            status = "⚠ FALLBACK"
            fallback_count += 1
        elif result.is_valid:
            valid_count += 1
            status = "✓ VALID"
        else:
            status = "✗ INVALID"

        print(f"       Status    : {status}")
        if result.response:
            print(f"       Confidence: {result.response.confidence.value}")
            print(f"       Sources   : {len(result.response.sources)} chunk(s)")
            print(f"       Answer    : {result.response.answer[:80]}...")
        if result.errors:
            for e in result.errors[-2:]:          # show last 2 errors
                print(f"       Error     : {e}")
        print(f"       Attempts  : {result.attempts}  |  Latency: {result.latency_ms:.0f}ms\n")

        results.append({
            "query_number"  : i,
            "query"         : query,
            "is_valid"      : result.is_valid,
            "used_fallback" : result.used_fallback,
            "attempts"      : result.attempts,
            "latency_ms"    : round(result.latency_ms, 1),
            "errors"        : result.errors,
            "response"      : result.response.model_dump() if result.response else None
        })

    # ── Metrics ───────────────────────────────────────────────────────────────
    total          = len(test_cases)
    valid_pct      = (valid_count / total) * 100
    fallback_pct   = (fallback_count / total) * 100
    avg_attempts   = total_attempts / total
    target_met     = valid_pct >= 95.0

    print("=" * 70)
    print("  FINAL METRICS")
    print("=" * 70)
    print(f"  Total queries      : {total}")
    print(f"  Valid (schema ok)  : {valid_count} / {total}  →  {valid_pct:.1f}%")
    print(f"  Fallbacks used     : {fallback_count} / {total}  →  {fallback_pct:.1f}%")
    print(f"  Avg attempts/query : {avg_attempts:.2f}")
    print(f"  95% target met     : {'✓ YES' if target_met else '✗ NO'}")
    print("=" * 70 + "\n")

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "run_timestamp"     : datetime.utcnow().isoformat() + "Z",
        "model"             : "llama-3.3-70b-versatile",
        "total_queries"     : total,
        "valid_count"       : valid_count,
        "fallback_count"    : fallback_count,
        "valid_percentage"  : round(valid_pct, 2),
        "target_95_met"     : target_met,
        "avg_attempts"      : round(avg_attempts, 2),
        "results"           : results
    }

    output_path = "test_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {output_path}\n")


if __name__ == "__main__":
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("ERROR: Set GROQ_API_KEY environment variable before running.")
        sys.exit(1)
    run_tests(api_key=key)