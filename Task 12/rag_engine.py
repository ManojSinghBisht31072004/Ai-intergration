"""
rag_engine.py
Chunking, Embedding (TF-IDF), and Vector Search
"""

import re
import math
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    char_start: int
    char_end: int
    token_estimate: int
    embedding: Optional[List[float]] = field(default=None, repr=False)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def rough_token_count(text: str) -> int:
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────────

class TextChunker:
    """
    Sliding-window chunker with sentence-boundary awareness.
    chunk_size    → target tokens per chunk
    chunk_overlap → how many tokens to repeat at chunk boundaries
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_sentences(self, text: str) -> List[Tuple[str, int]]:
        pattern = re.compile(r'(?<=[.!?])\s+')
        results = []
        prev = 0
        for m in pattern.finditer(text):
            s = text[prev:m.start() + 1].strip()
            if s:
                results.append((s, prev))
            prev = m.end()
        tail = text[prev:].strip()
        if tail:
            results.append((tail, prev))
        return results

    def chunk(self, text: str, doc_id: str, doc_title: str) -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks: List[Chunk] = []
        i = 0

        while i < len(sentences):
            buf_tokens = 0
            buf_sents: List[Tuple[str, int]] = []

            while i < len(sentences) and \
                  buf_tokens + rough_token_count(sentences[i][0]) <= self.chunk_size:
                buf_sents.append(sentences[i])
                buf_tokens += rough_token_count(sentences[i][0])
                i += 1

            if not buf_sents:          # single sentence > chunk_size
                buf_sents.append(sentences[i])
                i += 1

            chunk_text = " ".join(s for s, _ in buf_sents)
            char_start = buf_sents[0][1]
            char_end   = buf_sents[-1][1] + len(buf_sents[-1][0])
            chunk_id   = hashlib.md5(
                f"{doc_id}:{char_start}:{char_end}".encode()
            ).hexdigest()[:12]

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                token_estimate=buf_tokens,
            ))

            # Step back for overlap
            if self.chunk_overlap > 0 and i < len(sentences):
                overlap_tokens, step_back = 0, 0
                for sent, _ in reversed(buf_sents):
                    if overlap_tokens >= self.chunk_overlap:
                        break
                    overlap_tokens += rough_token_count(sent)
                    step_back += 1
                i -= step_back

        return chunks


# ─────────────────────────────────────────────
# EMBEDDER  (TF-IDF, no API needed)
# ─────────────────────────────────────────────

class LocalEmbedder:
    """
    Fit on the corpus once, then embed any text as a normalised TF-IDF vector.
    Cosine similarity between vectors = relevance score.
    """

    def __init__(self):
        self._vocab: Dict[str, int] = {}
        self._idf:   Dict[str, float] = {}
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]

    def fit(self, corpus: List[str]) -> None:
        N = len(corpus)
        df: Dict[str, int] = {}
        for doc in corpus:
            for t in set(self._tokenize(doc)):
                df[t] = df.get(t, 0) + 1
        self._vocab = {t: i for i, t in enumerate(sorted(df))}
        self._idf   = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
        self._fitted = True

    def _tf(self, tokens: List[str]) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = max(1, len(tokens))
        return {t: c / total for t, c in counts.items()}

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def embed(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        tokens = self._tokenize(text)
        tf     = self._tf(tokens)
        vec    = [0.0] * len(self._vocab)
        for t, tf_val in tf.items():
            if t in self._vocab:
                vec[self._vocab[t]] = tf_val * self._idf.get(t, 1.0)
        return self._normalize(vec)


# ─────────────────────────────────────────────
# VECTOR SEARCH
# ─────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))   # both already normalised


class VectorStore:
    """In-memory store: add chunks, search by query embedding."""

    def __init__(self):
        self._chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        scored = []
        for chunk in self._chunks:
            if chunk.embedding is None:
                continue
            score = cosine_similarity(query_embedding, chunk.embedding)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(chunk=c, score=s, rank=i + 1)
            for i, (s, c) in enumerate(scored[:top_k])
        ]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()