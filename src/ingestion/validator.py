"""Explicit validation pass over the loaded corpus. Pydantic already
enforces per-record shape at load time; this module checks
cross-collection referential integrity, which Pydantic alone cannot
express. Must be run — and pass — before embedding/indexing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas.decision import Decision
from src.schemas.document import Document
from src.schemas.evaluation import EvaluationQuery


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(self, message: str) -> None:
        self.errors.append(message)


def validate_corpus(
    decisions: list[Decision],
    documents: list[Document],
    queries: list[EvaluationQuery],
) -> ValidationReport:
    report = ValidationReport()

    decision_ids = {d.decision_id for d in decisions}
    doc_ids = {d.doc_id for d in documents}

    # Decisions -> documents
    for d in decisions:
        for eid in d.evidence_doc_ids:
            if eid not in doc_ids:
                report.add(f"{d.decision_id} references missing document {eid}")
        if d.preceding_decision_id and d.preceding_decision_id not in decision_ids:
            report.add(
                f"{d.decision_id} references missing preceding_decision_id "
                f"{d.preceding_decision_id}"
            )

    # Documents -> decisions
    for doc in documents:
        if doc.related_decision_id and doc.related_decision_id not in decision_ids:
            report.add(f"{doc.doc_id} references missing decision {doc.related_decision_id}")

    # Evaluation queries -> decisions/documents
    for q in queries:
        for did in q.expected_precedent_decision_ids + q.expected_context_decision_ids:
            if did not in decision_ids:
                report.add(f"{q.query_id} references missing decision {did}")
        for eid in q.expected_evidence_doc_ids:
            if eid not in doc_ids:
                report.add(f"{q.query_id} references missing document {eid}")

    # Chronological sanity: preceding_decision_id must actually precede in date
    by_id = {d.decision_id: d for d in decisions}
    for d in decisions:
        if d.preceding_decision_id:
            pred = by_id.get(d.preceding_decision_id)
            if pred and pred.date > d.date:
                report.add(
                    f"{d.decision_id} dated before its preceding_decision_id "
                    f"{d.preceding_decision_id}"
                )

    return report
