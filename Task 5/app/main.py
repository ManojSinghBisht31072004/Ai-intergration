"""
main.py
-------
FastAPI app exposing:
  POST /upload  — ingest a document
  POST /ask     — ask a question
  GET  /health  — check Gemini key rotation status
  GET  /docs    — auto Swagger UI (free from FastAPI)
"""

import os
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from app.ingestor import ingest_document
    from app.retriever import answer_question
    from app.gemini_client import key_manager
    from app.vectorstore import collection_count
except ModuleNotFoundError:
    from ingestor import ingest_document
    from retriever import answer_question
    from gemini_client import key_manager
    from vectorstore import collection_count

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Knowledge Base Assistant",
    description="Upload documents, ask questions, get cited answers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceItem(BaseModel):
    chunk_id: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    grounded: bool
    hallucination_detected: bool
    retrieved_chunks: int


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Returns app status and Gemini key rotation info."""
    return {
        "status": "ok",
        "total_chunks_in_db": collection_count(),
        "gemini_key_manager": key_manager.status(),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    The file is chunked, embedded, and stored in ChromaDB.
    """
    allowed = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use: {allowed}",
        )

    # save to temp file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=ext, dir=UPLOAD_DIR
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_document(tmp_path, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")
    finally:
        os.unlink(tmp_path)

    return {
        "message": "Document ingested successfully.",
        **result,
    }


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    """
    Ask a question. Returns a grounded answer with cited source chunks.
    If the answer isn't in the documents, returns 'I don't know...'.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(body.question, top_k=body.top_k)
    except RuntimeError as e:
        # all 5 keys failed
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

    return AskResponse(**result)