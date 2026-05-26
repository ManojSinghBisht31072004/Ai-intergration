import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import chromadb
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Setup ────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=api_key)

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("rag_docs")

app = FastAPI(title="Simple RAG API")


# ── Schema ────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


# ── Helpers ───────────────────────────────────────
def embed_query(text):
    result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=text,
    task_type="retrieval_query"
)
    return result["embedding"]


def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


# ── Routes ────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors_indexed": collection.count()
    }


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Run ingest.py first."
        )

    # Step 1 — embed the question
    q_vector = embed_query(req.question)

    # Step 2 — search Chroma
    results = collection.query(
        query_embeddings=[q_vector],
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "score": round(1 - dist, 4)
        })

    # Step 3 — build prompt
    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks
    )
    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {req.question}

Answer:"""

    # Step 4 — ask Gemini
    answer = ask_gemini(prompt)

    return {
        "answer": answer,
        "sources": list({c["source"] for c in chunks}),
        "chunks_used": len(chunks)
    }