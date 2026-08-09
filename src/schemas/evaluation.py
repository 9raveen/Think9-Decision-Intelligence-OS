"""Schema for evaluation queries — the ground truth used to score the
retrieval layer (and, in later phases, the agent layer) against known
correct behavior."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

QueryType = Literal["cross_brand_precedent", "no_precedent", "conflicting_precedent"]
CorrectBehavior = Literal[
    "surface_precedent_with_nuance", "no_precedent_found", "conflict_flag_human_review"
]


class EvaluationQuery(BaseModel):
    query_id: str = Field(..., pattern=r"^Q\d$")
    type: QueryType
    brand: str
    query_text: str
    expected_precedent_decision_ids: list[str] = Field(default_factory=list)
    expected_context_decision_ids: list[str] = Field(default_factory=list)
    expected_evidence_doc_ids: list[str] = Field(default_factory=list)
    expected_reasoning: str
    expected_outcome: str
    correct_behavior: CorrectBehavior
