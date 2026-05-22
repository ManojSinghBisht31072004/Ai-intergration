"""
main.py
Interactive RAG + Function Calling Demo
Day 12 — Run this file.
Supports: PDF, DOCX, TXT file uploads + manual paste + demo docs
"""

import os
import sys
import textwrap

from rag_engine import TextChunker, LocalEmbedder, VectorStore
from llm_client import run_rag_query


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TOP_K        = 4      # chunks to retrieve per query
CHUNK_SIZE   = 300    # tokens per chunk
OVERLAP      = 50     # overlap tokens


# ─────────────────────────────────────────────
# FILE READERS
# ─────────────────────────────────────────────

def read_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def read_pdf_file(filepath):
    try:
        import PyPDF2
        text = ""
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            print(f"   📃 Total pages found: {total_pages}")
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                    print(f"   ✔ Page {i+1} extracted ({len(extracted)} chars)")
                else:
                    print(f"   ⚠️  Page {i+1} — no text found (may be scanned image)")
        return text.strip()
    except ImportError:
        print("   ❌ PyPDF2 not installed. Run: pip install PyPDF2")
        return ""
    except Exception as e:
        print(f"   ❌ PDF read error: {e}")
        return ""


def read_docx_file(filepath):
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        print(f"   📃 Total paragraphs found: {len(paragraphs)}")
        return "\n".join(paragraphs)
    except ImportError:
        print("   ❌ python-docx not installed. Run: pip install python-docx")
        return ""
    except Exception as e:
        print(f"   ❌ DOCX read error: {e}")
        return ""


def load_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return read_pdf_file(filepath)
    elif ext == ".docx":
        return read_docx_file(filepath)
    elif ext in (".txt", ".md"):
        return read_txt_file(filepath)
    else:
        print(f"   ⚠️  Unsupported file type: {ext}")
        print(f"   Supported types: .pdf  .docx  .txt  .md")
        return ""


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def hr(char="─", width=65):
    print(char * width)


def banner():
    hr("═")
    print("  DAY 12 — RAG + FUNCTION CALLING")
    print("  Groq LLM  │  TF-IDF Vector Store  │  File Upload Support")
    hr("═")


def print_results(question, results, output):
    hr()
    print(f"❓ QUESTION: {question}")
    hr()
    print(f"📚 RETRIEVED CHUNKS ({len(results)}):")
    for r in results:
        snippet = r.chunk.text[:120].replace("\n", " ")
        print(f"   [{r.rank}] score={r.score:.3f} | Doc: '{r.chunk.doc_title}'")
        print(f"       {snippet}...")
    hr()
    if output["tools_called"]:
        print(f"🔧 TOOLS USED ({len(output['tools_called'])}):")
        for t in output["tools_called"]:
            print(f"   • {t['name']}({t['arguments']})")
            snippet = str(t["result"])[:120]
            print(f"     → {snippet}")
        hr()
    else:
        print("🔧 TOOLS USED: None (RAG context was sufficient)")
        hr()
    print("💬 ANSWER:")
    wrapped = textwrap.fill(
        output["answer"],
        width=65,
        initial_indent="   ",
        subsequent_indent="   "
    )
    print(wrapped)
    hr("═")


def print_loaded_docs(documents):
    hr()
    print(f"📂 DOCUMENTS LOADED ({len(documents)}):")
    for i, doc in enumerate(documents, 1):
        chars  = len(doc["text"])
        tokens = chars // 4
        print(f"   [{i}] {doc['title']}")
        print(f"       {chars} characters  │  ~{tokens} tokens")
    hr()


# ─────────────────────────────────────────────
# DEMO DOCUMENTS
# ─────────────────────────────────────────────

def get_demo_documents():
    return [
        {
            "title": "Introduction to RAG Systems",
            "text": """
Retrieval-Augmented Generation (RAG) is a powerful AI framework that combines the strengths
of retrieval-based and generative approaches. In a RAG system, when a user asks a question,
the system first searches a knowledge base for relevant documents or passages. These retrieved
passages are then provided as context to a large language model, which generates a final answer.

The main components of a RAG pipeline include: a document store or corpus, a text chunking
strategy to split documents into manageable pieces, an embedding model to convert text into
vector representations, a vector database to store and search embeddings, and a language model
to generate answers from the retrieved context.

RAG significantly reduces hallucinations compared to pure generation because the model is
grounded in actual retrieved evidence. It also keeps knowledge up-to-date without retraining
the model, since you can simply update the document store. RAG is widely used in enterprise
Q&A systems, customer support bots, and research assistants.

Chunking strategy is critical in RAG. Chunks that are too small lose context; chunks that are
too large dilute relevance. A chunk size of 256-512 tokens with 10-20% overlap is a common
starting point. Sentence-aware splitting preserves semantic coherence better than character-level splitting.

Embedding models convert text to dense vectors where semantic similarity is captured by
vector proximity. Popular models include OpenAI text-embedding-3, Sentence Transformers,
and Cohere embeddings. For fast local inference, TF-IDF or BM25 can serve as lightweight alternatives.
            """.strip()
        },
        {
            "title": "Function Calling in LLMs",
            "text": """
Function calling, also known as tool use, is a capability in modern large language models
that allows them to request the execution of external tools or functions. Instead of generating
a free-text answer, the model can output a structured JSON object specifying a function name
and its arguments, which the host application then executes.

The workflow for function calling is as follows. First, you define one or more tools with a
name, description, and parameter schema in JSON Schema format. You pass these tool definitions
to the LLM along with the user's message. If the model determines a tool is needed, it outputs
a tool_call object. Your code executes the tool and sends the result back as a tool message.
The model then generates a final answer incorporating the tool result.

Groq supports OpenAI-compatible function calling via the tools parameter. You set tool_choice
to auto to let the model decide when to call a tool, or force a specific tool by name. The model
can call multiple tools in sequence or in parallel depending on the task.

Common use cases for tool calling include: web search for up-to-date information, database
queries, mathematical calculations, code execution, weather lookups, calendar access, and
retrieval from secondary knowledge bases. When combined with RAG, function calling creates
a flexible system where the model can both use retrieved context and call tools as needed.

Best practices for tool design: write clear, specific descriptions so the model knows when
to use each tool; use strict parameter schemas to avoid malformed calls; always validate and
sanitize tool inputs before execution; log all tool calls for debugging and auditing.
            """.strip()
        },
        {
            "title": "Groq API and LLM Inference",
            "text": """
Groq is an AI infrastructure company that provides ultra-fast LLM inference through its
Language Processing Unit (LPU) technology. The Groq API is compatible with the OpenAI SDK,
making it easy to switch from OpenAI to Groq by changing the base URL and API key.

To use the Groq Python client, install it with pip install groq. Initialize the client with
your API key: client = Groq(api_key="your-key"). Call client.chat.completions.create() with
model, messages, and optional tools parameters. Supported models include llama-3.3-70b-versatile,
llama3-8b-8192, mixtral-8x7b-32768, and gemma-7b-it.

Groq's key advantage is speed: it delivers inference at hundreds of tokens per second, much
faster than traditional GPU-based providers. This makes it ideal for interactive applications,
real-time chatbots, and agentic workflows with multiple LLM calls.

The API supports streaming responses, system prompts, conversation history, JSON mode,
function calling, and temperature control. Rate limits vary by plan: the free tier allows
a limited number of requests per minute and tokens per day.

When using function calling on Groq, pass the tools array of JSON schema objects and set
tool_choice to auto or a specific function. The response will include a tool_calls field
if the model wants to invoke a tool. Parse the JSON arguments, execute the function locally,
and append a tool role message with the result before calling the API again.
            """.strip()
        },
        {
            "title": "Python Best Practices for AI Projects",
            "text": """
Building robust AI projects in Python requires attention to code structure, dependency management,
and error handling. Use virtual environments (venv or conda) to isolate project dependencies.
Store API keys in environment variables, never hardcode them in source files. Use python-dotenv
to load .env files automatically.

Project structure matters. Separate concerns into modules: data ingestion, embedding, retrieval,
LLM calls, and tools. Use dataclasses or Pydantic models for structured data. Write type hints
for all function signatures to catch errors early with mypy or pyright.

For LLM projects specifically, always implement retry logic with exponential backoff for API calls,
since rate limits and transient errors are common. Cache embeddings to disk to avoid recomputing
them on every run. For large document sets, use batch embedding calls instead of one at a time.

Logging is essential for debugging agentic systems. Log every LLM call with the model, token count,
and latency. Log every tool call with inputs and outputs. Use Python's built-in logging module
with rotating file handlers for production deployments.

Testing AI pipelines requires both unit tests and integration tests. Use pytest with fixtures
for reusable test data. Mock LLM API calls in unit tests to avoid costs and flakiness.
Measure retrieval quality with recall@k and MRR metrics on a labeled evaluation set.
            """.strip()
        },
    ]


# ─────────────────────────────────────────────
# DOCUMENT INPUT — FILE UPLOAD
# ─────────────────────────────────────────────

def get_documents_from_files():
    print("\n" + "─" * 65)
    print("  📁 FILE UPLOAD MODE")
    print("─" * 65)
    print("  Supported formats: PDF  |  DOCX  |  TXT  |  MD")
    print("  Enter full file path, or drag-and-drop into terminal.")
    print("  Type DONE when finished.\n")

    documents = []

    while True:
        raw = input("  File path (or DONE): ").strip().strip('"').strip("'")

        if raw.upper() == "DONE":
            break

        if not raw:
            continue

        if not os.path.exists(raw):
            print(f"   ❌ File not found: {raw}")
            print(f"      Check the path and try again.\n")
            continue

        filename = os.path.basename(raw)
        title    = os.path.splitext(filename)[0]
        ext      = os.path.splitext(filename)[1].lower()

        print(f"\n   📄 Loading: {filename}  [{ext}]")
        text = load_file(raw)

        if not text:
            print(f"   ❌ No text extracted from '{filename}'. Skipping.\n")
            continue

        # Let user optionally rename the document title
        custom_title = input(f"   Title for this doc (Enter to keep '{title}'): ").strip()
        if custom_title:
            title = custom_title

        documents.append({"title": title, "text": text})
        print(f"   ✅ '{title}' loaded — {len(text):,} chars  │  ~{len(text)//4:,} tokens\n")

    return documents


# ─────────────────────────────────────────────
# DOCUMENT INPUT — MANUAL PASTE
# ─────────────────────────────────────────────

def get_documents_from_paste():
    print("\n" + "─" * 65)
    print("  ✏️  MANUAL PASTE MODE")
    print("─" * 65)
    print("  Enter a title then paste your text.")
    print("  Type END on its own line when done with each document.")
    print("  Type DONE as the title when you have no more documents.\n")

    documents = []

    while True:
        title = input("  Document title (or DONE): ").strip()
        if title.upper() == "DONE":
            break
        if not title:
            continue

        print(f"  Paste text for '{title}' — type END on a new line to finish:")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)

        text = "\n".join(lines).strip()
        if text:
            documents.append({"title": title, "text": text})
            print(f"   ✅ '{title}' added — {len(text):,} chars  │  ~{len(text)//4:,} tokens\n")
        else:
            print(f"   ⚠️  No text entered for '{title}'. Skipping.\n")

    return documents


# ─────────────────────────────────────────────
# DOCUMENT INPUT — MAIN MENU
# ─────────────────────────────────────────────

def get_user_documents():
    print("\n" + "═" * 65)
    print("  STEP 1 — LOAD YOUR DOCUMENTS")
    print("═" * 65)
    print("  [1]  Upload files    (PDF / DOCX / TXT / MD)")
    print("  [2]  Paste text      (type or paste directly)")
    print("  [3]  Demo documents  (built-in sample content)")
    print("═" * 65)

    while True:
        choice = input("\n  Enter 1, 2 or 3: ").strip()

        if choice == "1":
            docs = get_documents_from_files()
            if not docs:
                print("\n  ⚠️  No files loaded. Please try again or choose another option.")
                continue
            return docs

        elif choice == "2":
            docs = get_documents_from_paste()
            if not docs:
                print("\n  ⚠️  No documents entered. Please try again or choose another option.")
                continue
            return docs

        elif choice == "3":
            print("\n  ✅ Loading built-in demo documents...")
            return get_demo_documents()

        else:
            print("  ❌ Invalid choice. Enter 1, 2 or 3.")


# ─────────────────────────────────────────────
# INGEST PIPELINE
# ─────────────────────────────────────────────

def ingest_documents(chunker, embedder, store, documents):
    print("\n" + "═" * 65)
    print("  STEP 2 — INGESTING & INDEXING")
    print("═" * 65)

    all_chunks = []

    for i, doc in enumerate(documents):
        doc_id = f"doc_{i}"
        chunks = chunker.chunk(doc["text"], doc_id=doc_id, doc_title=doc["title"])
        all_chunks.extend(chunks)
        print(f"  ✔ '{doc['title']}' → {len(chunks)} chunks")

    print(f"\n  🔢 Building TF-IDF embeddings for {len(all_chunks)} chunks...")
    corpus = [c.text for c in all_chunks]
    embedder.fit(corpus)

    for chunk in all_chunks:
        chunk.embedding = embedder.embed(chunk.text)

    store.add(all_chunks)
    print(f"  ✅ Vector store ready — {store.count()} chunks indexed.")
    print("═" * 65)
    return all_chunks


# ─────────────────────────────────────────────
# QUERY PIPELINE
# ─────────────────────────────────────────────

def ask(question, embedder, store, verbose=True):
    query_embedding = embedder.embed(question)
    results         = store.search(query_embedding, top_k=TOP_K)
    output          = run_rag_query(
        question=question,
        context_chunks=results,
        api_key=GROQ_API_KEY,
        verbose=verbose,
    )
    return results, output


# ─────────────────────────────────────────────
# HELP MENU
# ─────────────────────────────────────────────

def print_help():
    hr()
    print("  COMMANDS:")
    print("   quit / exit / q  →  Exit the program")
    print("   reset            →  Load new documents")
    print("   docs             →  Show loaded documents")
    print("   help             →  Show this menu")
    print()
    print("  EXAMPLE QUESTIONS (Data Science docs):")
    print("   • What is supervised learning?")
    print("   • Explain the bias-variance tradeoff")
    print("   • What are steps in data preprocessing?")
    print("   • What is overfitting and how to prevent it?")
    print("   • What is today's date?")
    print("   • Calculate accuracy if 90 out of 120 are correct")
    print("   • Tell me more about neural networks in detail")
    print("   • What are latest trends in data science 2025?")
    hr()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    banner()

    # API key check
    if GROQ_API_KEY == "your-groq-api-key-here":
        print("\n  ⚠️  GROQ_API_KEY not set!")
        print("  Set it before running:\n")
        print("  Windows :  set GROQ_API_KEY=gsk_xxxxxxxxxxxx")
        print("  Mac/Linux: export GROQ_API_KEY=gsk_xxxxxxxxxxxx\n")
        sys.exit(1)

    # Init components
    chunker  = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    embedder = LocalEmbedder()
    store    = VectorStore()

    # Load documents
    documents = get_user_documents()
    print_loaded_docs(documents)

    # Ingest
    ingest_documents(chunker, embedder, store, documents)

    # Ready
    print("\n  ✅ Ready! Ask anything about your documents.")
    print("  Type 'help' for commands and example questions.\n")

    # Q&A loop
    while True:
        hr()
        try:
            question = input("❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break

        if not question:
            continue

        # Commands
        if question.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!")
            break

        if question.lower() == "reset":
            store.clear()
            embedder.__init__()
            documents = get_user_documents()
            print_loaded_docs(documents)
            ingest_documents(chunker, embedder, store, documents)
            print("\n  ✅ Ready!\n")
            continue

        if question.lower() == "docs":
            print_loaded_docs(documents)
            continue

        if question.lower() == "help":
            print_help()
            continue

        # Run RAG + tool pipeline
        print()
        try:
            results, output = ask(question, embedder, store)
            print_results(question, results, output)
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            print("  Check your API key and internet connection.\n")


if __name__ == "__main__":
    main()