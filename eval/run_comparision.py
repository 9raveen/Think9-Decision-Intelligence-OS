"""Runs Q1-Q7 through BOTH the TF-IDF baseline and the semantic/Qdrant
retriever, using identical methodology (same K, same relevance threshold,
same metric functions) so the comparison is fair.

Usage:
    python eval/run_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import conflict_retrieval_success, no_precedent_false_positive, recall_at_k
from src.ingestion.embed import TextEmbedder
from src.ingestion.loader import load_corpus
from src.ingestion.semantic_embed import SemanticEmbedder
from src.ingestion.validator import validate_corpus
from src.retrieval.decision_retriever import DecisionRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qdrant_store import QdrantDocumentStore
from src.retrieval.vector_store import DocumentVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RELEVANCE_THRESHOLD = 0.12  # unchanged from run_eval.py — same threshold used
                            # for both pipelines so the comparison is fair
K = 6


def build_pipeline(embedder_kind: str, decisions, documents) -> HybridRetriever:
    if embedder_kind == "tfidf":
        decision_retriever = DecisionRetriever(embedder=TextEmbedder())
        decision_retriever.index(decisions)
        document_store = DocumentVectorStore(embedder=TextEmbedder())
        document_store.upsert(documents)
    elif embedder_kind == "semantic":
        decision_retriever = DecisionRetriever(embedder=SemanticEmbedder())
        decision_retriever.index(decisions)
        document_store = QdrantDocumentStore(embedder=SemanticEmbedder())
        document_store.upsert(documents, decisions)
    else:
        raise ValueError(embedder_kind)
    return HybridRetriever(decision_retriever, document_store)


def run(hybrid: HybridRetriever, queries) -> dict[str, dict]:
    out = {}
    for q in queries:
        bundle = hybrid.retrieve(q.query_text, decision_top_k=K, document_top_k=K)
        retrieved_decision_ids = bundle.decision_ids()
        retrieved_document_ids = bundle.document_ids()
        top_decision_score = bundle.matched_decisions[0].score if bundle.matched_decisions else 0.0

        row = {"top_decision_score": round(top_decision_score, 4)}
        if q.type == "cross_brand_precedent":
            row["decision_recall@6"] = round(
                recall_at_k(retrieved_decision_ids, q.expected_precedent_decision_ids, K), 3
            )
            row["evidence_recall@6"] = round(
                recall_at_k(retrieved_document_ids, q.expected_evidence_doc_ids, K), 3
            )
        elif q.type == "no_precedent":
            row["false_positive"] = no_precedent_false_positive(top_decision_score, RELEVANCE_THRESHOLD)
        elif q.type == "conflicting_precedent":
            row["conflict_success"] = conflict_retrieval_success(
                retrieved_decision_ids, q.expected_precedent_decision_ids, K
            )
        out[q.query_id] = row
    return out


def main() -> None:
    decisions, documents, queries = load_corpus(DATA_DIR)
    report = validate_corpus(decisions, documents, queries)
    if not report.is_valid:
        print("VALIDATION FAILED:", report.errors)
        sys.exit(1)

    tfidf_hybrid = build_pipeline("tfidf", decisions, documents)
    semantic_hybrid = build_pipeline("semantic", decisions, documents)

    tfidf_results = run(tfidf_hybrid, queries)
    semantic_results = run(semantic_hybrid, queries)

    print(f"{'Query':<6}{'TF-IDF Dec Recall':<20}{'Semantic Dec Recall':<22}"
          f"{'TF-IDF Ev Recall':<19}{'Semantic Ev Recall':<21}{'FP / Conflict'}")
    print("-" * 110)
    comparison = []
    for q in queries:
        t, s = tfidf_results[q.query_id], semantic_results[q.query_id]
        if q.type == "cross_brand_precedent":
            fp_conflict = "-"
            row = (
                f"{q.query_id:<6}{t['decision_recall@6']:<20}{s['decision_recall@6']:<22}"
                f"{t['evidence_recall@6']:<19}{s['evidence_recall@6']:<21}{fp_conflict}"
            )
        elif q.type == "no_precedent":
            fp_conflict = f"TFIDF_FP={t['false_positive']} SEM_FP={s['false_positive']}"
            row = f"{q.query_id:<6}{'-':<20}{'-':<22}{'-':<19}{'-':<21}{fp_conflict}"
        else:
            fp_conflict = f"TFIDF={t['conflict_success']} SEM={s['conflict_success']}"
            row = f"{q.query_id:<6}{'-':<20}{'-':<22}{'-':<19}{'-':<21}{fp_conflict}"
        print(row)
        comparison.append({"query_id": q.query_id, "type": q.type, "tfidf": t, "semantic": s})

    out_path = Path(__file__).resolve().parent / "results" / "comparison_eval.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nFull comparison written to {out_path}")


if __name__ == "__main__":
    main()
