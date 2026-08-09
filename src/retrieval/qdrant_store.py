"""Document vector store backed by Qdrant (embedded/local mode — no
server, no network required, verified working via QdrantClient(':memory:')).

Same upsert/search interface as DocumentVectorStore (src/retrieval/
vector_store.py) so HybridRetriever can take either one interchangeably.
The difference is the embedding used (semantic/LSA here vs TF-IDF there)
and that vectors + metadata are stored in Qdrant rather than an in-process
numpy array — closer to how this would run in production.

Metadata payload per point includes both the document's own fields and,
where the document is linked to a Decision (related_decision_id), that
decision's brand/function/supplier/product_line/category/tags — joined in
so future metadata-aware filtering (e.g. "only legal-function evidence")
doesn't require a second lookup.
"""
from __future__ import annotations

from typing import Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.ingestion.semantic_embed import SemanticEmbedder
from src.retrieval.vector_store import ScoredDocument
from src.schemas.decision import Decision
from src.schemas.document import Document

COLLECTION_NAME = "think9_documents"


class QdrantDocumentStore:
    def __init__(self, embedder: SemanticEmbedder | None = None) -> None:
        self._embedder = embedder or SemanticEmbedder()
        self._client = QdrantClient(":memory:")
        self._documents: list[Document] = []
        self._id_to_document: dict[str, Document] = {}

    def upsert(self, documents: list[Document], decisions: Optional[list[Decision]] = None) -> None:
        self._documents = documents
        decisions_by_id = {d.decision_id: d for d in (decisions or [])}

        texts = [d.searchable_text() for d in documents]
        self._embedder.fit(texts)
        vectors = self._embedder.transform(texts)

        if self._client.collection_exists(COLLECTION_NAME):
            self._client.delete_collection(COLLECTION_NAME)
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=vectors.shape[1], distance=qmodels.Distance.COSINE),
        )

        points = []
        for doc, vec in zip(documents, vectors):
            linked_decision = decisions_by_id.get(doc.related_decision_id) if doc.related_decision_id else None
            point_id = str(uuid4())
            self._id_to_document[point_id] = doc
            payload = {
                "doc_id": doc.doc_id,
                "document_type": doc.type,
                "date": doc.date.isoformat(),
                "brand": doc.brand,
                "related_decision_id": doc.related_decision_id,
                "function": linked_decision.function if linked_decision else None,
                "supplier": linked_decision.supplier_tags() if linked_decision else [],
                "product_line": linked_decision.product_line if linked_decision else None,
                "category": linked_decision.product_or_category if linked_decision else None,
                "tags": linked_decision.tags if linked_decision else [],
            }
            points.append(qmodels.PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

        self._client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query_text: str, top_k: int = 5) -> list[ScoredDocument]:
        query_vec = self._embedder.transform([query_text])[0]
        response = self._client.query_points(collection_name=COLLECTION_NAME, query=query_vec.tolist(), limit=top_k)
        return [ScoredDocument(document=self._id_to_document[hit.id], score=float(hit.score)) for hit in response.points]

    def get_payload(self, doc_id: str) -> Optional[dict]:
        records, _ = self._client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]),
            limit=1,
        )
        return records[0].payload if records else None