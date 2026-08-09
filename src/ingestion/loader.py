"""Loads the raw JSON corpus into Pydantic models. Loading and
validation are kept separate: this module only parses; validator.py
checks referential integrity across the loaded collections."""
from __future__ import annotations

import json
from pathlib import Path

from src.schemas.decision import Decision
from src.schemas.document import Document
from src.schemas.evaluation import EvaluationQuery


def load_decisions(path: Path) -> list[Decision]:
    raw = json.loads(Path(path).read_text())
    return [Decision.model_validate(item) for item in raw]


def load_documents(path: Path) -> list[Document]:
    raw = json.loads(Path(path).read_text())
    return [Document.model_validate(item) for item in raw]


def load_evaluation_queries(path: Path) -> list[EvaluationQuery]:
    raw = json.loads(Path(path).read_text())
    return [EvaluationQuery.model_validate(item) for item in raw]


def load_corpus(data_dir: Path) -> tuple[list[Decision], list[Document], list[EvaluationQuery]]:
    data_dir = Path(data_dir)
    decisions = load_decisions(data_dir / "decisions.json")
    documents = load_documents(data_dir / "documents.json")
    queries = load_evaluation_queries(data_dir / "evaluation_queries.json")
    return decisions, documents, queries
