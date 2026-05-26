import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import chromadb
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Setup ────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=api_key)

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("rag_docs")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


# ── Helpers ──────────────────────────────────────
def make_id(source, index):
    return hashlib.md5(f"{source}__{index}".encode()).hexdigest()


def embed(texts):
    result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=texts,
    task_type="retrieval_document"
)
    return result["embedding"]


def load_file(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext in (".txt", ".md"):
        return TextLoader(path).load()
    else:
        print(f"Skipping unsupported file: {path}")
        return []


# ── Main ─────────────────────────────────────────
def ingest(data_dir="./data"):
    files = list(Path(data_dir).glob("**/*"))
    files = [f for f in files if f.suffix.lower() in (".pdf", ".txt", ".md")]

    if not files:
        print("No files found in ./data — add some PDFs or .txt files first.")
        return

    for fpath in files:
        print(f"\nIndexing: {fpath.name}")
        docs = load_file(str(fpath))

        if not docs:
            continue

        chunks = splitter.split_documents(docs)
        print(f"  Split into {len(chunks)} chunks")

        texts = [c.page_content for c in chunks]
        ids = [make_id(str(fpath), i) for i in range(len(chunks))]
        metadatas = [
            {
                "source": fpath.name,
                "page": c.metadata.get("page", 0)
            }
            for c in chunks
        ]

        # embed in batches of 50
        all_embeddings = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            print(f"  Embedding batch {i // 50 + 1}...")
            all_embeddings.extend(embed(batch))

        collection.upsert(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"  ✅ {len(chunks)} chunks indexed from {fpath.name}")

    print(f"\n✅ Done. Total vectors in store: {collection.count()}")


if __name__ == "__main__":
    ingest()