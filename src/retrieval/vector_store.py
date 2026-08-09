"""In-memory vector store for document semantic search.

Interface is deliberately Qdrant-shaped (`upsert`, `search`) so
swapping in a real Qdrant collection later only requires reimplementing
this class, not any caller of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ingestion.embed import TextEmbedder
from src.schemas.document import Document


@dataclass
class ScoredDocument:
    document: Document
    score: float


class DocumentVectorStore:
    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self._embedder = embedder or TextEmbedder()
        self._documents: list[Document] = []
        self._vectors: np.ndarray | None = None

    def upsert(self, documents: list[Document]) -> None:
        self._documents = documents
        texts = [d.searchable_text() for d in documents]
        self._embedder.fit(texts)
        self._vectors = self._embedder.transform(texts)

    def search(self, query_text: str, top_k: int = 5) -> list[ScoredDocument]:
        if self._vectors is None:
            raise RuntimeError("upsert() must be called before search()")
        query_vec = self._embedder.transform([query_text])[0]
        scores = self._embedder.similarity(query_vec, self._vectors)
        ranked_idx = np.argsort(-scores)[:top_k]
        return [
            ScoredDocument(document=self._documents[i], score=float(scores[i]))
            for i in ranked_idx
        ]
    