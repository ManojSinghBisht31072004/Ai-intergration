"""
test_cases.py
Run all 10 test cases automatically and print a report.
Usage: python test_cases.py
"""

import os
import sys
from rag_engine import TextChunker, LocalEmbedder, VectorStore
from llm_client import run_rag_query
from main import ingest_documents, get_demo_documents

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

TEST_CASES = [
    # (id, question, expects_tool, expected_tool_name or None, notes)
    (1,  "What is RAG and what are its main components?",
         False, None,
         "Pure RAG — answer fully in demo docs"),

    (2,  "How does function calling work in LLMs?",
         False, None,
         "Pure RAG — covered in demo docs"),

    (3,  "What is the Groq API and what makes it fast?",
         False, None,
         "Pure RAG — Groq doc covers this"),

    (4,  "What Python libraries should I use for an AI project?",
         False, None,
         "Pure RAG — Python best practices doc"),

    (5,  "What chunk size is recommended for RAG?",
         False, None,
         "Pure RAG — specific detail in RAG doc"),

    (6,  "What is today's date?",
         True, "get_current_date",
         "Tool required — not in documents"),

    (7,  "If I have 1500 tokens and each chunk is 300 tokens with 50 overlap, how many chunks will I get?",
         True, "calculate",
         "Tool required — math calculation"),

    (8,  "Tell me more about machine learning in detail.",
         True, "fetch_additional_doc",
         "Tool required — not covered in docs"),

    (9,  "What are the latest developments in vector databases in 2025?",
         True, "search_web_summary",
         "Tool required — recent/external info"),

    (10, "Explain what chunking is and also calculate 256 times 4.",
         True, "calculate",
         "Hybrid — RAG for chunking, tool for math"),
]


def run_tests():
    print("=" * 70)
    print("  DAY 12 — RAG + FUNCTION CALLING: AUTOMATED TEST SUITE")
    print("=" * 70)

    # Setup
    chunker  = TextChunker(chunk_size=300, chunk_overlap=50)
    embedder = LocalEmbedder()
    store    = VectorStore()
    docs     = get_demo_documents()
    ingest_documents(chunker, embedder, store, docs)

    results_summary = []

    for (tid, question, expects_tool, expected_tool, notes) in TEST_CASES:
        print(f"\n{'─'*70}")
        print(f"TEST {tid:02d}: {question}")
        print(f"         Expects tool: {expects_tool} ({expected_tool or 'None'})")
        print(f"         Notes: {notes}")
        print()

        # Retrieve
        q_emb    = embedder.embed(question)
        chunks   = store.search(q_emb, top_k=4)
        output   = run_rag_query(question, chunks, GROQ_API_KEY, verbose=True)

        tools_used    = [t["name"] for t in output["tools_called"]]
        tool_was_used = len(tools_used) > 0
        correct_tool  = (expected_tool in tools_used) if expected_tool else True

        # Determine pass/fail
        expectation_met = (tool_was_used == expects_tool)
        passed = expectation_met and correct_tool

        print(f"\n  ANSWER (truncated): {output['answer'][:200]}...")
        print(f"  Tools called: {tools_used or 'None'}")
        print(f"  Expected tool used: {expects_tool} | Actual: {tool_was_used}")
        print(f"  RESULT: {'✅ PASS' if passed else '❌ FAIL'}")

        results_summary.append({
            "id":           tid,
            "question":     question[:50],
            "expects_tool": expects_tool,
            "tool_used":    tool_was_used,
            "tools":        tools_used,
            "passed":       passed,
        })

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'ID':<4} {'Question':<42} {'Tool?':<6} {'Pass'}")
    print(f"  {'─'*4} {'─'*42} {'─'*6} {'─'*4}")
    for r in results_summary:
        status = "✅" if r["passed"] else "❌"
        tool   = "Yes" if r["tool_used"] else "No"
        print(f"  {r['id']:<4} {r['question']:<42} {tool:<6} {status}")

    passed_count = sum(1 for r in results_summary if r["passed"])
    print(f"\n  TOTAL: {passed_count}/{len(TEST_CASES)} passed")
    print("=" * 70)


if __name__ == "__main__":
    if GROQ_API_KEY == "your-groq-api-key-here":
        print("Set GROQ_API_KEY environment variable first.")
        sys.exit(1)
    run_tests()