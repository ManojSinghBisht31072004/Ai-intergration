"""
chunker.py
----------
Splits a document string into overlapping chunks.
Each chunk gets a unique ID: {doc_id}_chunk_{n}
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_hint: int = 0   # rough page number if available


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """
    Split text into overlapping character-level chunks.
    chunk_size=500 chars, overlap=80 chars.
    """
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{index}",
                    doc_id=doc_id,
                    text=chunk_text_str,
                    page_hint=index,
                )
            )
            index += 1

        start = end - overlap  # slide window with overlap

    return chunks