"""Schema for structured Decision records.

A Decision represents a single documented business decision made by a
Think9 brand — what problem triggered it, what was decided, why, and
what evidence backs it. This is the core unit the retrieval layer
matches against when looking for cross-brand precedent.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


Function = Literal["procurement", "legal", "product", "quality", "operations"]
Scope = Literal["brand_specific", "portfolio_relevant"]
ProductLine = Literal["PET_packaging", "flexible_packaging", "glass_packaging"]


class Decision(BaseModel):
    decision_id: str = Field(..., pattern=r"^D\d{2}$")
    brand: str
    function: Function
    date: date
    problem: str
    options_considered: list[str]
    decision_made: str
    reason: str
    evidence_doc_ids: list[str] = Field(default_factory=list)
    owner: str
    outcome: str
    tags: list[str] = Field(default_factory=list)

    product_or_category: str
    scope: Scope
    review_required: bool
    product_line: Optional[ProductLine] = None
    preceding_decision_id: Optional[str] = None

    def searchable_text(self) -> str:
        return " ".join(
            [
                self.problem,
                " ".join(self.options_considered),
                self.decision_made,
                self.reason,
                self.outcome,
                self.product_or_category,
            ]
        )

    def supplier_tags(self) -> list[str]:
        return [t.split(":", 1)[1] for t in self.tags if t.startswith("supplier:")]