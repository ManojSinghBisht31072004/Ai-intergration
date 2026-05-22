"""
tools.py
Tool schemas (sent to LLM) + execution functions (run locally)
"""

import json
import datetime
import math
from typing import Dict, Any


# ─────────────────────────────────────────────
# TOOL SCHEMAS  (what the LLM sees)
# ─────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_additional_doc",
            "description": (
                "Fetch supplementary reference content on a topic when the "
                "retrieved chunks don't fully answer the question. Use this "
                "when the user asks about something not covered in context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or concept to fetch extra info about."
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["brief", "detailed"],
                        "description": "How much detail to return."
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform a mathematical calculation. Use when the user asks "
                "for numeric computation, statistics, or math."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe Python math expression, e.g. '200 * 0.15' or 'sqrt(144)'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Return today's date and time. Use when the user asks about current date/time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web_summary",
            "description": (
                "Simulate a web search and return a summary for the query. "
                "Use when the user asks about something recent or external "
                "that is not in the provided document context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ─────────────────────────────────────────────
# TOOL EXECUTION  (runs locally)
# ─────────────────────────────────────────────

def fetch_additional_doc(topic: str, detail_level: str = "brief") -> str:
    """
    Simulated document fetch.
    In production: hit a DB, API, or second vector store.
    """
    library = {
        "machine learning": {
            "brief": "Machine learning is a subset of AI where models learn patterns from data.",
            "detailed": (
                "Machine learning (ML) is a branch of artificial intelligence that enables systems "
                "to learn and improve from experience without being explicitly programmed. "
                "Core types: supervised learning (labeled data), unsupervised learning (clustering, "
                "dimensionality reduction), and reinforcement learning (reward signals). "
                "Common algorithms: linear regression, decision trees, SVMs, neural networks, "
                "gradient boosting. Key concepts: training/test split, overfitting, regularization, "
                "cross-validation, hyperparameter tuning."
            )
        },
        "rag": {
            "brief": "RAG (Retrieval-Augmented Generation) combines vector search with LLM generation.",
            "detailed": (
                "Retrieval-Augmented Generation (RAG) is an AI framework that enhances LLM responses "
                "by retrieving relevant documents from a knowledge base before generation. "
                "Pipeline: query → embed → vector search → top-k chunks → prompt context → LLM → answer. "
                "Benefits: reduces hallucination, keeps knowledge up-to-date without retraining, "
                "allows citation of sources. Key components: chunking strategy, embedding model, "
                "vector database (FAISS, Pinecone, Chroma), retrieval (dense, sparse, hybrid), reranking."
            )
        },
        "function calling": {
            "brief": "Function calling lets LLMs request execution of predefined tools.",
            "detailed": (
                "Function calling (tool use) allows LLMs to generate structured JSON requests "
                "to invoke external tools. The LLM decides when a tool is needed, outputs a "
                "tool_call with name + arguments, the host application executes it, and the "
                "result is passed back as a tool message. Enables: web search, database queries, "
                "calculations, API calls, code execution. Groq supports OpenAI-compatible tool "
                "calling with the tools parameter."
            )
        },
        "python": {
            "brief": "Python is a high-level, interpreted programming language.",
            "detailed": (
                "Python is a versatile, high-level language known for readability and a huge ecosystem. "
                "Key features: dynamic typing, first-class functions, list comprehensions, generators, "
                "decorators, context managers. Popular libraries: NumPy/Pandas (data), "
                "Scikit-learn/PyTorch/TensorFlow (ML), FastAPI/Flask (web), Requests (HTTP). "
                "Package manager: pip. Virtual envs: venv or conda."
            )
        }
    }

    topic_lower = topic.lower()
    for key, content in library.items():
        if key in topic_lower:
            return content.get(detail_level, content["brief"])

    return (
        f"No specific supplementary document found for '{topic}'. "
        f"Based on general knowledge: {topic} is a concept worth exploring further "
        f"through authoritative sources."
    )


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    safe_globals = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
        "min": min,
        "max": max,
        "sum": sum,
    }
    try:
        result = eval(expression, safe_globals)   # noqa: S307
        return f"Result of `{expression}` = {result}"
    except Exception as exc:
        return f"Calculation error: {exc}"


def get_current_date() -> str:
    now = datetime.datetime.now()
    return (
        f"Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')} "
        f"(Timezone: local system time)"
    )


def search_web_summary(query: str) -> str:
    """
    Simulated web search.
    In production: call SerpAPI, Tavily, Brave Search, etc.
    """
    return (
        f"[Simulated web search for: '{query}']\n"
        f"Top result summary: This topic is actively discussed online. "
        f"Recent sources suggest that '{query}' involves multiple perspectives "
        f"and ongoing developments. For accurate real-time data, integrate a "
        f"live search API (Tavily, SerpAPI, Brave) in production."
    )


# ─────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────

def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Route a tool call to its implementation."""
    dispatch = {
        "fetch_additional_doc": fetch_additional_doc,
        "calculate":            calculate,
        "get_current_date":     get_current_date,
        "search_web_summary":   search_web_summary,
    }
    if name not in dispatch:
        return f"Error: Unknown tool '{name}'"
    try:
        return dispatch[name](**arguments)
    except Exception as exc:
        return f"Tool '{name}' error: {exc}"