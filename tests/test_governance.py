"""Real tests for src/governance/rules.py.

Replaces the previous stub, which skipped every test with the reason
"Governance rules not yet implemented" — stale, since governance is
fully implemented and wired into the pipeline. These tests exercise
all three trigger rules independently, plus the conflict-trigger and
the previously-fixed "must only scan post-threshold relevant decisions"
regression described in rules.py's own docstring.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.governance.rules import evaluate
from src.schemas.decision import Decision


def _make_decision(**overrides) -> Decision:
    defaults = dict(
        decision_id="D01",
        brand="Nova",
        function="procurement",
        date=date(2025, 1, 1),
        problem="Test problem",
        options_considered=["A", "B"],
        decision_made="Chose A",
        reason="Because reasons",
        evidence_doc_ids=[],
        owner="test-owner",
        outcome="Worked fine",
        tags=[],
        product_or_category="test-category",
        scope="brand_specific",
        review_required=False,
    )
    defaults.update(overrides)
    return Decision(**defaults)


def test_no_triggers_means_no_review():
    d = _make_decision()
    result = evaluate([d], has_conflict=False)
    assert result.review_required is False
    assert result.reasons == []


def test_evidence_level_review_flag_triggers_review():
    d = _make_decision(review_required=True)
    result = evaluate([d], has_conflict=False)
    assert result.review_required is True
    assert any("flagged review_required" in r for r in result.reasons)


def test_high_stakes_function_legal_triggers_review():
    d = _make_decision(function="legal", review_required=False)
    result = evaluate([d], has_conflict=False)
    assert result.review_required is True
    assert any("high-stakes function" in r for r in result.reasons)


def test_high_stakes_function_quality_triggers_review():
    d = _make_decision(function="quality", review_required=False)
    result = evaluate([d], has_conflict=False)
    assert result.review_required is True
    assert any("high-stakes function" in r for r in result.reasons)


def test_non_high_stakes_function_does_not_trigger_alone():
    d = _make_decision(function="procurement", review_required=False, scope="brand_specific")
    result = evaluate([d], has_conflict=False)
    assert result.review_required is False


def test_portfolio_relevant_scope_triggers_review():
    d = _make_decision(scope="portfolio_relevant", review_required=False)
    result = evaluate([d], has_conflict=False)
    assert result.review_required is True
    assert any("portfolio-wide relevance" in r for r in result.reasons)


def test_conflict_flag_triggers_review_independent_of_decision_fields():
    d = _make_decision(review_required=False, function="procurement", scope="brand_specific")
    result = evaluate([d], has_conflict=True)
    assert result.review_required is True
    assert any("disagree on outcome" in r for r in result.reasons)


def test_multiple_triggers_all_reported_without_duplicates():
    d1 = _make_decision(decision_id="D01", review_required=True, function="legal")
    d2 = _make_decision(decision_id="D02", review_required=True, function="legal")
    result = evaluate([d1, d2], has_conflict=True)
    assert result.review_required is True
    # 2 decisions x 2 per-decision reasons + 1 conflict reason = 5, no dupes
    assert len(result.reasons) == len(set(result.reasons)) == 5


def test_empty_relevant_decisions_and_no_conflict_means_no_review():
    result = evaluate([], has_conflict=False)
    assert result.review_required is False
    assert result.reasons == []


def test_only_scans_decisions_it_is_given_not_a_hidden_global_set():
    """Regression guard for the bug documented in rules.py's own
    docstring: governance must only evaluate decisions the caller passes
    in (i.e. the Cross-Brand Agent's post-threshold relevant set), never
    reach into some other global/raw retrieval set on its own."""
    irrelevant_but_flagged = _make_decision(decision_id="D99", review_required=True)
    # Caller (pipeline) is responsible for only passing relevant decisions;
    # evaluate() itself has no other data source it could pull from.
    result_with = evaluate([irrelevant_but_flagged], has_conflict=False)
    result_without = evaluate([], has_conflict=False)
    assert result_with.review_required is True
    assert result_without.review_required is False