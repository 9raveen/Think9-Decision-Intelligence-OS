"""Simple, honest evaluation metrics for the retrieval layer.

These measure retrieval quality only — whether the right evidence
surfaces — not whether a downstream agent reasons about it correctly.
That's a deliberate scope boundary matching the retrieval/reasoning
split in the architecture.
"""
from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Fraction of expected_ids present in the top-k retrieved_ids.
    Returns 1.0 if expected_ids is empty (nothing to recall)."""
    if not expected_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k)
    return hits / len(expected_ids)


def no_precedent_false_positive(
    top_decision_score: float, relevance_threshold: float
) -> bool:
    """True if a no-precedent query produced a decision score above the
    relevance threshold — i.e. the retriever would mislead a downstream
    agent into thinking a precedent exists where the ground truth says
    none does."""
    return top_decision_score >= relevance_threshold


def conflict_retrieval_success(
    retrieved_decision_ids: list[str], expected_conflicting_ids: list[str], k: int
) -> bool:
    """True only if ALL expected conflicting decisions are present in
    the top-k — a conflict can't be flagged downstream if only one side
    of it was ever retrieved."""
    top_k = set(retrieved_decision_ids[:k])
    return all(did in top_k for did in expected_conflicting_ids)