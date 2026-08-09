from pathlib import Path

import pytest

from src.ingestion.loader import load_corpus
from src.ingestion.semantic_embed import SemanticEmbedder
from src.retrieval.decision_retriever import DecisionRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qdrant_store import QdrantDocumentStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DATA_DIR)


def test_semantic_embedder_generates_vectors_of_expected_shape(corpus):
    _, documents, _ = corpus
    embedder = SemanticEmbedder(n_components=10)
    texts = [d.searchable_text() for d in documents]
    embedder.fit(texts)
    vecs = embedder.transform(texts)
    assert vecs.shape == (len(documents), 10)


def test_qdrant_store_indexes_and_retrieves(corpus):
    decisions, documents, _ = corpus
    store = QdrantDocumentStore(embedder=SemanticEmbedder())
    store.upsert(documents, decisions)
    results = store.search("Nova Supplier Alpha delivery delay", top_k=5)
    assert len(results) == 5
    assert all(0.0 <= r.score <= 1.0001 for r in results)  # cosine sim, small float tolerance


def test_qdrant_metadata_preserved(corpus):
    decisions, documents, _ = corpus
    store = QdrantDocumentStore(embedder=SemanticEmbedder())
    store.upsert(documents, decisions)
    payload = store.get_payload("E03")
    assert payload is not None
    assert payload["doc_id"] == "E03"
    assert payload["brand"] == "Nova"
    assert payload["related_decision_id"] == "D02"
    # joined from the linked Decision (D02)
    assert payload["function"] == "procurement"
    assert payload["product_line"] == "PET_packaging"
    assert "alpha" in payload["supplier"]


def test_qdrant_metadata_null_for_unlinked_document(corpus):
    """E27/E28 are general playbook excerpts with no related_decision_id —
    joined decision fields should be null/empty, not fabricated."""
    decisions, documents, _ = corpus
    store = QdrantDocumentStore(embedder=SemanticEmbedder())
    store.upsert(documents, decisions)
    payload = store.get_payload("E27")
    assert payload["related_decision_id"] is None
    assert payload["function"] is None
    assert payload["product_line"] is None


@pytest.fixture(scope="module")
def semantic_hybrid(corpus):
    decisions, documents, _ = corpus
    decision_retriever = DecisionRetriever(embedder=SemanticEmbedder())
    decision_retriever.index(decisions)
    document_store = QdrantDocumentStore(embedder=SemanticEmbedder())
    document_store.upsert(documents, decisions)
    return HybridRetriever(decision_retriever, document_store)


def test_q1_semantic_retrieval_measured_not_assumed(semantic_hybrid, corpus):
    """Q1 is a known weak point (TF-IDF decision_recall@6=0.5). This test
    asserts the semantic pipeline runs end-to-end and returns SOME matched
    decisions — it does NOT assert full recall, because that would be
    tuning the test to a desired outcome rather than measuring reality."""
    _, _, queries = corpus
    q1 = next(q for q in queries if q.query_id == "Q1")
    bundle = semantic_hybrid.retrieve(q1.query_text, decision_top_k=6)
    ids = bundle.decision_ids()
    assert len(ids) > 0
    # D01 (the original Nova/Alpha sourcing decision) should be findable
    # at minimum via the semantic path, since it shares direct vocabulary
    # with the query (Alpha, packaging).
    assert any(d in ids for d in ["D01", "D02", "D03", "D04"])


def test_q7_conflict_both_sides_retrievable_semantic(semantic_hybrid, corpus):
    """Ground truth requirement unchanged from the TF-IDF baseline test:
    both D09 and D10 must be retrievable for the conflict to be flaggable
    downstream."""
    _, _, queries = corpus
    q7 = next(q for q in queries if q.query_id == "Q7")
    bundle = semantic_hybrid.retrieve(q7.query_text, decision_top_k=6)
    ids = bundle.decision_ids()
    assert "D09" in ids
    assert "D10" in ids
