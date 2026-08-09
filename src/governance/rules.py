"""Deterministic governance rules.

Deliberately NOT an LLM call — whether a recommendation needs human
sign-off is a business rule, not a judgment call to delegate to a model.
Every rule here is a plain boolean check over fields already on the
Decision records or the retrieval outcome — auditable and predictable.

IMPORTANT: this takes the Cross-Brand Agent's *relevant* decisions
(post-threshold), not the raw retrieval bundle. A bug during development
had this scanning ALL raw top-k retrieval hits, including weak/irrelevant
ones below the relevance threshold — which meant a no-precedent query
could still trigger "human review required" based on an unrelated
decision's fields. Fixed: governance now only looks at decisions the
Cross-Brand Agent actually judged relevant.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.schemas.decision import Decision

HIGH_STAKES_FUNCTIONS = {"legal", "quality"}


@dataclass
class GovernanceResult:
    review_required: bool
    reasons: list[str]


def evaluate(relevant_decisions: list[Decision], has_conflict: bool) -> GovernanceResult:
    reasons: list[str] = []

    for d in relevant_decisions:
        if d.review_required:
            reasons.append(f"{d.decision_id} is flagged review_required in its own record")
        if d.function in HIGH_STAKES_FUNCTIONS:
            reasons.append(f"{d.decision_id} is a {d.function} decision (high-stakes function)")
        if d.scope == "portfolio_relevant":
            reasons.append(f"{d.decision_id} has portfolio-wide relevance, not brand-local")

    if has_conflict:
        reasons.append("retrieved precedents disagree on outcome — conflict requires human judgment")

    seen: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.append(r)

    return GovernanceResult(review_required=bool(seen), reasons=seen)