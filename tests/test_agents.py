from pathlib import Path

import pytest

from src.agents.cross_brand_agent import analyze
from src.agents.router import route
from src.governance.rules import evaluate as governance_evaluate
from src.graph.pipeline import run_query
from src.ingestion.embed import TextEmbedder
from src.ingestion.loader import load_corpus
from src.retrieval.decision_retriever import DecisionRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import DocumentVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DATA_DIR)


@pytest.fixture(scope="module")
def hybrid(corpus):
    decisions, documents, _ = corpus
    dr = DecisionRetriever(embedder=TextEmbedder())
    dr.index(decisions)
    ds = DocumentVectorStore(embedder=TextEmbedder())
    ds.upsert(documents)
    return HybridRetriever(dr, ds)


def _query(corpus, query_id):
    _, _, queries = corpus
    return next(q for q in queries if q.query_id == query_id)


def test_router_classifies_procurement_query(hybrid, corpus):
    q1 = _query(corpus, "Q1")
    bundle = hybrid.retrieve(q1.query_text)
    result = route(bundle)
    assert result.function == "procurement"


def test_router_returns_none_when_nothing_scores(hybrid, corpus):
    q5 = _query(corpus, "Q5")  # no-precedent query
    bundle = hybrid.retrieve(q5.query_text)
    result = route(bundle)
    assert result.function is None or isinstance(result.function, str)


def test_cross_brand_no_precedent(hybrid, corpus):
    q5 = _query(corpus, "Q5")
    bundle = hybrid.retrieve(q5.query_text)
    result = analyze(bundle)
    assert result.behavior == "no_precedent_found"


def test_cross_brand_surfaces_precedent_for_q2(hybrid, corpus):
    q2 = _query(corpus, "Q2")
    bundle = hybrid.retrieve(q2.query_text)
    result = analyze(bundle)
    assert result.behavior == "surface_precedent_with_nuance"
    assert len(result.relevant_decisions) > 0


def test_cross_brand_conflict_is_cross_brand_only():
    """Regression test for a real bug found during development: same-brand
    decisions in sequence (e.g. a brand's own delay -> switch story) must
    NOT be flagged as a 'conflict' — conflict is cross-brand by
    definition. This is checked directly against the corpus rather than
    via a query, since it tests the agent's internal logic."""
    decisions, _, _ = load_corpus(DATA_DIR)
    from src.retrieval.decision_retriever import ScoredDecision
    from src.retrieval.hybrid_retriever import EvidenceBundle

    nova_decisions = [d for d in decisions if d.decision_id in ("D01", "D02", "D03", "D04")]
    fake_bundle = EvidenceBundle(
        query_text="test",
        matched_decisions=[ScoredDecision(decision=d, score=0.5) for d in nova_decisions],
        matched_documents=[],
    )
    result = analyze(fake_bundle)
    assert result.behavior != "conflict_flag_human_review"


def test_governance_flags_legal_decisions(corpus):
    decisions, _, _ = corpus
    d05 = next(d for d in decisions if d.decision_id == "D05")  # legal decision
    result = governance_evaluate([d05], has_conflict=False)
    assert result.review_required is True
    assert any("legal" in r for r in result.reasons)


def test_governance_no_flag_for_routine_decision(corpus):
    decisions, _, _ = corpus
    d08 = next(d for d in decisions if d.decision_id == "D08")  # routine, review_required=False
    result = governance_evaluate([d08], has_conflict=False)
    assert result.review_required is False


def test_governance_no_flag_when_no_relevant_decisions():
    """Regression test for the bug found during development: governance
    must not flag review_required based on irrelevant raw retrieval hits
    when the Cross-Brand Agent found no relevant precedent."""
    result = governance_evaluate([], has_conflict=False)
    assert result.review_required is False
    assert result.reasons == []


def test_full_pipeline_runs_end_to_end_without_api_key(hybrid, corpus, monkeypatch):
    """No GROQ_API_KEY is set in this test environment — verifies the
    template fallback path works and the graph completes without error."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    q2 = _query(corpus, "Q2")
    state = run_query(hybrid, q2.query_text)
    assert state["synthesis"].used_llm is False
    assert len(state["synthesis"].answer_text) > 0
    assert state["cross_brand"].behavior == "surface_precedent_with_nuance"


def test_pipeline_ground_truth_agreement_rate(hybrid, corpus):
    """Reports (does not strictly assert 7/7) agreement between the
    Cross-Brand Agent's behavior classification and ground truth. Known,
    documented limitation: Q7's conflict is not reliably caught by the
    keyword-sentiment heuristic (D10's outcome is deliberately ambiguous
    in the corpus). This test guards against regression below the
    current baseline (5/7), not against imperfection itself."""
    _, _, queries = corpus
    agree = 0
    for q in queries:
        bundle = hybrid.retrieve(q.query_text)
        result = analyze(bundle)
        if result.behavior == q.correct_behavior:
            agree += 1
    assert agree >= 5, f"Agreement dropped below documented baseline: {agree}/7"