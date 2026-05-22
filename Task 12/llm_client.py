"""
llm_client.py
Groq API interaction — RAG context injection + tool-use loop
"""

import json
import os
from typing import List, Dict, Any, Optional

from groq import Groq

from tools import TOOL_SCHEMAS, execute_tool


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"   # fast + capable on Groq
MAX_TOOL_ROUNDS = 5          # prevent infinite loops


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def build_system_prompt() -> str:
    return (
        "You are a knowledgeable assistant with access to a document knowledge base. "
        "You will be given retrieved context chunks from the user's documents. "
        "Always ground your answers in the provided context first. "
        "If the context is insufficient, use the available tools to fetch more information. "
        "If a calculation is needed, use the calculate tool. "
        "If the user asks about today's date/time, use get_current_date. "
        "If you need external or supplementary information, use fetch_additional_doc or search_web_summary. "
        "Be concise, accurate, and always cite which document/chunk your answer comes from when possible."
    )


def build_user_message(question: str, context_chunks: list) -> str:
    if not context_chunks:
        context_block = "No relevant chunks retrieved from the knowledge base."
    else:
        parts = []
        for i, result in enumerate(context_chunks, 1):
            parts.append(
                f"[Chunk {i} | Doc: '{result.chunk.doc_title}' | "
                f"Score: {result.score:.3f}]\n{result.chunk.text}"
            )
        context_block = "\n\n".join(parts)

    return (
        f"RETRIEVED CONTEXT FROM KNOWLEDGE BASE:\n"
        f"{'='*60}\n"
        f"{context_block}\n"
        f"{'='*60}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"Answer based on the context above. Use tools if needed."
    )


# ─────────────────────────────────────────────
# TOOL-USE LOOP
# ─────────────────────────────────────────────

def run_rag_query(
    question: str,
    context_chunks: list,
    api_key: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full RAG + function-calling loop.

    Returns:
        {
          "answer":        str,
          "tools_called":  list of {name, arguments, result},
          "rounds":        int,
          "messages":      full message history
        }
    """
    client = Groq(api_key=api_key)

    messages: List[Dict] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": build_user_message(question, context_chunks)},
    ]

    tools_called = []
    rounds = 0

    while rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        if verbose:
            print(f"  [LLM Round {rounds}] Calling Groq...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )

        msg = response.choices[0].message

        # ── No tool call → final answer ──────────────────────────────
        if not msg.tool_calls:
            if verbose:
                print(f"  [LLM Round {rounds}] Final answer received (no tool call).")
            return {
                "answer":       msg.content,
                "tools_called": tools_called,
                "rounds":       rounds,
                "messages":     messages,
            }

        # ── Tool call(s) → execute → feed result back ─────────────────
        # Append assistant message with tool_calls
        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ]
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            if verbose:
                print(f"  [Tool Called] {name}({arguments})")

            result = execute_tool(name, arguments)

            if verbose:
                print(f"  [Tool Result] {result[:120]}{'...' if len(result) > 120 else ''}")

            tools_called.append({
                "name":      name,
                "arguments": arguments,
                "result":    result,
            })

            # Append tool result message
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    # Max rounds hit — force final answer
    if verbose:
        print(f"  [LLM] Max tool rounds reached. Forcing final answer.")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    return {
        "answer":       response.choices[0].message.content,
        "tools_called": tools_called,
        "rounds":       rounds,
        "messages":     messages,
    }