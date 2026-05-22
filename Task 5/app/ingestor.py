"""
ingestor.py
-----------
Full ingestion pipeline:
  1. Parse text from uploaded file (PDF or TXT)
  2. Chunk the text
  3. Embed each chunk via Gemini (with key rotation)
  4. Store in ChromaDB
"""

import re
import uuid
import logging
from pathlib import Path

from pypdf import PdfReader

try:
    from app.chunker import chunk_text
    from app.vectorstore import store_chunks
    from app.gemini_client import embed_text
except ModuleNotFoundError:
    from chunker import chunk_text
    from vectorstore import store_chunks
    from gemini_client import embed_text

logger = logging.getLogger(__name__)


def _extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _clean_text(text: str) -> str:
    # collapse whitespace, remove non-printable chars
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def ingest_document(file_path: str, filename: str) -> dict:
    """
    Full pipeline: parse → clean → chunk → embed → store.
    Returns a summary dict.
    """
    ext = Path(filename).suffix.lower()
    doc_id = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}"

    logger.info(f"Ingesting '{filename}' as doc_id={doc_id}")

    # 1. Extract text
    if ext == ".pdf":
        raw_text = _extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        raw_text = _extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")

    text = _clean_text(raw_text)
    if len(text) < 50:
        raise ValueError("Document appears to be empty or unreadable.")

    # 2. Chunk
    chunks = chunk_text(text, doc_id=doc_id, chunk_size=500, overlap=80)
    logger.info(f"Created {len(chunks)} chunks")

    # 3. Embed — each call goes through key rotation automatically
    embeddings = []
    for i, chunk in enumerate(chunks):
        vec = embed_text(chunk.text)
        embeddings.append(vec)
        if (i + 1) % 10 == 0:
            logger.info(f"  Embedded {i + 1}/{len(chunks)} chunks...")

    # 4. Store
    stored = store_chunks(chunks, embeddings)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_chunks": stored,
        "text_length": len(text),
    }