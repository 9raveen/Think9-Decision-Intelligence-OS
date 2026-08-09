"""Combines Path A (structured decision retrieval) and Path B (document
vector search) into a single evidence bundle for a query.

IMPORTANT ARCHITECTURAL BOUNDARY: this module returns *evidence* —
ranked decisions and documents with scores. It does not decide whether
that evidence constitutes a "relevant precedent," does not flag
conflicts, and does not generate an answer. That reasoning belongs to
the Cross-Brand Intelligence Agent (a later phase, not yet built). No
LLM calls happen here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.decision_retriever import DecisionRetriever, ScoredDecision
from src.retrieval.vector_store import DocumentVectorStore, ScoredDocument


@dataclass
class EvidenceBundle:
    query_text: str
    matched_decisions: list[ScoredDecision]
    matched_documents: list[ScoredDocument]
    # documents pulled in because they're evidence_doc_ids of a matched
    # decision, even if they didn't independently score highly
    linked_documents: list[ScoredDocument] = field(default_factory=list)

    def decision_ids(self, min_score: float = 0.0) -> list[str]:
        return [sd.decision.decision_id for sd in self.matched_decisions if sd.score >= min_score]

    def document_ids(self) -> list[str]:
        seen: list[str] = []
        for sd in self.matched_documents + self.linked_documents:
            if sd.document.doc_id not in seen:
                seen.append(sd.document.doc_id)
        return seen


class HybridRetriever:
    def __init__(
        self,
        decision_retriever: DecisionRetriever,
        document_store: DocumentVectorStore,
    ) -> None:
        self._decision_retriever = decision_retriever
        self._document_store = document_store
        self._doc_by_id = {d.doc_id: d for d in document_store._documents}

    def retrieve(
        self,
        query_text: str,
        decision_top_k: int = 6,
        document_top_k: int = 6,
    ) -> EvidenceBundle:
        matched_decisions = self._decision_retriever.semantic_search(query_text, top_k=decision_top_k)
        matched_documents = self._document_store.search(query_text, top_k=document_top_k)

        # Pull in documents explicitly linked as evidence to top-scoring
        # decisions, so evidence isn't lost just because a document's own
        # text scored lower than the top_k cutoff.
        linked_ids: set[str] = set()
        for sd in matched_decisions:
            linked_ids.update(sd.decision.evidence_doc_ids)
        already_included = {sd.document.doc_id for sd in matched_documents}
        linked_documents = [
            ScoredDocument(document=self._doc_by_id[doc_id], score=0.0)
            for doc_id in linked_ids
            if doc_id in self._doc_by_id and doc_id not in already_included
        ]

        return EvidenceBundle(
            query_text=query_text,
            matched_decisions=matched_decisions,
            matched_documents=matched_documents,
            linked_documents=linked_documents,
        )
