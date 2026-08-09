"""Full pipeline eval. Usage: python eval/run_agent_eval.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.pipeline import run_query
from src.ingestion.embed import TextEmbedder
from src.ingestion.loader import load_corpus
from src.ingestion.validator import validate_corpus
from src.retrieval.decision_retriever import DecisionRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import DocumentVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    decisions, documents, queries = load_corpus(DATA_DIR)
    report = validate_corpus(decisions, documents, queries)
    if not report.is_valid:
        print("VALIDATION FAILED:", report.errors)
        sys.exit(1)

    decision_retriever = DecisionRetriever(embedder=TextEmbedder())
    decision_retriever.index(decisions)
    document_store = DocumentVectorStore(embedder=TextEmbedder())
    document_store.upsert(documents)
    hybrid = HybridRetriever(decision_retriever, document_store)

    print(f"{'Query':<6}{'Expected':<28}{'Got':<28}{'Match':<7}{'Governance'}")
    print("-" * 100)
    agree = 0
    for q in queries:
        state = run_query(hybrid, q.query_text)
        got = state["cross_brand"].behavior
        match = "OK" if got == q.correct_behavior else "MISMATCH"
        agree += got == q.correct_behavior
        gov = "REVIEW" if state["governance"].review_required else "-"
        print(f"{q.query_id:<6}{q.correct_behavior:<28}{got:<28}{match:<7}{gov}")

    print(f"\nGround-truth agreement: {agree}/{len(queries)}")


if __name__ == "__main__":
    main()