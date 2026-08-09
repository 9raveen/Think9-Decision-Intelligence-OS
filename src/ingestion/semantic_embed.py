"""Semantic embedding — LSA (TF-IDF + Truncated SVD), NOT a transformer model.

WHY NOT A REAL SENTENCE-TRANSFORMER: this sandbox's network allowlist does
not include huggingface.co (verified: HTTP 403 on direct request), which is
where sentence-transformers models are hosted. `pip install
sentence-transformers` succeeds, but `SentenceTransformer("all-MiniLM-L6-v2")`
would fail at model-download time. Rather than silently fall back to TF-IDF
again and call it "semantic," this module implements Latent Semantic
Analysis (TF-IDF -> Truncated SVD) as a genuinely different, dense embedding
that captures term co-occurrence structure — it can match texts that share
no exact vocabulary but co-occur with the same latent topics elsewhere in
the corpus (e.g. "shipment arrived late" and "delivery delay" can end up
close if the corpus has enough shared context). It will NOT do what a real
transformer embedding does (deep paraphrase/synonym understanding from
pretraining on billions of tokens) — with a corpus this small (46 texts),
LSA's latent topics are also inherently limited by how much co-occurrence
signal exists. Report results accordingly, don't oversell this as
transformer-quality semantic search.

Swap path for a real deployment: replace this class with a
sentence-transformers or hosted-embedding-API client implementing the same
fit/transform/similarity interface — no caller code changes required.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline


class SemanticEmbedder:
    def __init__(self, n_components: int = 40) -> None:
        self._n_components = n_components
        self._pipeline = None
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        n_components = min(self._n_components, len(texts) - 1)
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._pipeline = make_pipeline(tfidf, svd)
        self._pipeline.fit(texts)
        self._fitted = True

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SemanticEmbedder must be fit() before transform()")
        return self._pipeline.transform(texts)

    def similarity(self, query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        return cosine_similarity(query_vec, corpus_vecs)[0]