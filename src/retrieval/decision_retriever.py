"""Path A: structured retrieval over Decision records.

This is distinct from document vector search (Path B) because a
Decision has structured fields (brand, supplier, product_line, tags)
that support precise filtering *before* any semantic ranking — a
capability free-text document search doesn't have. Decisions carry
their own semantic score too (via TF-IDF over searchable_text), so a
decision can be found either by explicit filter or by narrative
similarity to the query.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from src.ingestion.embed import TextEmbedder
from src.schemas.decision import Decision


@dataclass
class ScoredDecision:
    decision: Decision
    score: float


class DecisionRetriever:
    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self._embedder = embedder or TextEmbedder()
        self._decisions: list[Decision] = []
        self._vectors: np.ndarray | None = None

    def index(self, decisions: list[Decision]) -> None:
        self._decisions = decisions
        texts = [d.searchable_text() for d in decisions]
        self._embedder.fit(texts)
        self._vectors = self._embedder.transform(texts)

    def filter(
        self,
        brand: str | None = None,
        function: str | None = None,
        supplier: str | None = None,
        product_line: str | None = None,
        tags: list[str] | None = None,
        after: date | None = None,
        before: date | None = None,
    ) -> list[Decision]:
        results = self._decisions
        if brand:
            results = [d for d in results if d.brand.lower() == brand.lower()]
        if function:
            results = [d for d in results if d.function == function]
        if supplier:
            results = [d for d in results if supplier.lower() in d.supplier_tags()]
        if product_line:
            results = [d for d in results if d.product_line == product_line]
        if tags:
            wanted = set(tags)
            results = [d for d in results if wanted.issubset(set(d.tags))]
        if after:
            results = [d for d in results if d.date >= after]
        if before:
            results = [d for d in results if d.date <= before]
        return results

    def semantic_search(self, query_text: str, top_k: int = 10) -> list[ScoredDecision]:
        if self._vectors is None:
            raise RuntimeError("index() must be called before semantic_search()")
        query_vec = self._embedder.transform([query_text])[0]
        scores = self._embedder.similarity(query_vec, self._vectors)
        ranked_idx = np.argsort(-scores)[:top_k]
        return [
            ScoredDecision(decision=self._decisions[i], score=float(scores[i]))
            for i in ranked_idx
        ]
    