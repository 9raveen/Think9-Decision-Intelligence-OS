from pathlib import Path

import pytest

from src.ingestion.loader import load_corpus
from src.ingestion.validator import validate_corpus
from src.retrieval.decision_retriever import DecisionRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import DocumentVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DATA_DIR)


def test_corpus_validates(corpus):
    decisions, documents, queries = corpus
    report = validate_corpus(decisions, documents, queries)
    assert report.is_valid, report.errors


def test_decision_retriever_filters_by_supplier(corpus):
    decisions, _, _ = corpus
    retriever = DecisionRetriever()
    retriever.index(decisions)
    alpha_decisions = retriever.filter(supplier="alpha")
    ids = {d.decision_id for d in alpha_decisions}
    assert {"D01", "D02", "D03", "D04", "D08", "D14", "D17"}.issubset(ids)


def test_decision_retriever_filters_by_product_line(corpus):
    decisions, _, _ = corpus
    retriever = DecisionRetriever()
    retriever.index(decisions)
    pet_decisions = retriever.filter(product_line="PET_packaging")
    ids = {d.decision_id for d in pet_decisions}
    assert ids == {"D01", "D02", "D03", "D04"}


@pytest.fixture(scope="module")
def hybrid(corpus):
    decisions, documents, _ = corpus
    decision_retriever = DecisionRetriever()
    decision_retriever.index(decisions)
    document_store = DocumentVectorStore()
    document_store.upsert(documents)
    return HybridRetriever(decision_retriever, document_store)


def test_q7_conflict_both_sides_retrievable(hybrid, corpus):
    """Ground truth: Q7 requires BOTH D09 (Kindle positive Beta) and D10
    (Verve Beta incident) to be retrievable — a conflict can't be flagged
    if only one side surfaces."""
    _, _, queries = corpus
    q7 = next(q for q in queries if q.query_id == "Q7")
    bundle = hybrid.retrieve(q7.query_text, decision_top_k=6)
    ids = bundle.decision_ids()
    assert "D09" in ids
    assert "D10" in ids


def test_hybrid_returns_evidence_not_a_verdict(hybrid, corpus):
    """Architectural boundary check: EvidenceBundle must not contain any
    reasoning/verdict field — only ranked decisions and documents."""
    _, _, queries = corpus
    q1 = next(q for q in queries if q.query_id == "Q1")
    bundle = hybrid.retrieve(q1.query_text)
    bundle_fields = set(vars(bundle).keys())
    assert bundle_fields == {"query_text", "matched_decisions", "matched_documents", "linked_documents"}
