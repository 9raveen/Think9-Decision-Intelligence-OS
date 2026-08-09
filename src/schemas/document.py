"""Schema for unstructured supporting evidence documents (meeting notes,
emails, QA reports, legal memos, playbook excerpts, etc.)."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str = Field(..., pattern=r"^E\d{2}$")
    type: str
    date: date
    brand: Optional[str] = None
    related_decision_id: Optional[str] = None
    content: str

    def searchable_text(self) -> str:
        return self.content
