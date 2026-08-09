"""Runs Q1-Q7 through the retrieval layer and reports metrics against
the approved ground truth in data/evaluation_queries.json.

Usage:
    python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import conflict_retrieval_success, no_precedent_false_positive, recall_at_k
from src.ingestion.loader import load_corpus
from src.ingestion.validator import validate_corpus
from src.retrieval.decision_retriever import DecisionRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import DocumentVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RELEVANCE_THRESHOLD = 0.12  # TF-IDF cosine score above which a decision is
                            # treated as a plausible match. Tuned empirically
                            # for this MVP corpus/embedder; will need
                            # re-tuning if swapped to a real embedding model.
K = 6


def main() -> None:
    decisions, documents, queries = load_corpus(DATA_DIR)

    report = validate_corpus(decisions, documents, queries)
    if not report.is_valid:
        print("VALIDATION FAILED:")
        for err in report.errors:
            print(" -", err)
        sys.exit(1)
    print(f"Validation passed: {len(decisions)} decisions, {len(documents)} documents, "
          f"{len(queries)} queries.\n")

    decision_retriever = DecisionRetriever()
    decision_retriever.index(decisions)

    document_store = DocumentVectorStore()
    document_store.upsert(documents)

    hybrid = HybridRetriever(decision_retriever, document_store)

    results = []
    for q in queries:
        bundle = hybrid.retrieve(q.query_text, decision_top_k=K, document_top_k=K)
        retrieved_decision_ids = bundle.decision_ids()
        retrieved_document_ids = bundle.document_ids()
        top_decision_score = bundle.matched_decisions[0].score if bundle.matched_decisions else 0.0

        row = {
            "query_id": q.query_id,
            "type": q.type,
            "correct_behavior": q.correct_behavior,
            "top_decision_score": round(top_decision_score, 4),
            "retrieved_decision_ids_top6": retrieved_decision_ids,
        }

        if q.type == "cross_brand_precedent":
            row["decision_recall@6"] = round(
                recall_at_k(retrieved_decision_ids, q.expected_precedent_decision_ids, K), 3
            )
            row["evidence_recall@6"] = round(
                recall_at_k(retrieved_document_ids, q.expected_evidence_doc_ids, K), 3
            )
        elif q.type == "no_precedent":
            row["false_positive"] = no_precedent_false_positive(
                top_decision_score, RELEVANCE_THRESHOLD
            )
        elif q.type == "conflicting_precedent":
            row["conflict_retrieval_success"] = conflict_retrieval_success(
                retrieved_decision_ids, q.expected_precedent_decision_ids, K
            )

        results.append(row)

    out_path = Path(__file__).resolve().parent / "results" / "retrieval_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"{'Query':<5} {'Type':<22} {'Metric':<28} {'Value'}")
    print("-" * 75)
    for row in results:
        if row["type"] == "cross_brand_precedent":
            metric = f"decision_recall={row['decision_recall@6']} evidence_recall={row['evidence_recall@6']}"
        elif row["type"] == "no_precedent":
            metric = f"false_positive={row['false_positive']} (score={row['top_decision_score']})"
        else:
            metric = f"conflict_success={row['conflict_retrieval_success']}"
        print(f"{row['query_id']:<5} {row['type']:<22} {metric}")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
