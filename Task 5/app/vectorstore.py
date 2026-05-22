"""
vectorstore.py
--------------
ChromaDB wrapper.
Stores: chunk_id, embedding, raw text, doc_id
Retrieves: top-k most similar chunks for a query embedding
"""

import chromadb
from chromadb.config import Settings
from app.chunker import Chunk

# persist to disk so data survives restarts
_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False),
)

COLLECTION_NAME = "knowledge_base"


def _get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    """
    Upsert chunks into ChromaDB.
    Returns number of chunks stored.
    """
    collection = _get_collection()

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"doc_id": c.doc_id, "page_hint": c.page_hint}
            for c in chunks
        ],
    )

    return len(chunks)


def retrieve_top_k(
    query_embedding: list[float],
    k: int = 5,
) -> list[dict]:
    """
    Return the top-k most similar chunks.
    Each result: { chunk_id, text, doc_id, distance }
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i, chunk_id in enumerate(results["ids"][0]):
        output.append(
            {
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                "distance": round(results["distances"][0][i], 4),
            }
        )

    return output


def collection_count() -> int:
    return _get_collection().count()


def reset_collection():
    """Wipe the collection — useful for testing."""
    _client.delete_collection(COLLECTION_NAME)