"""Text -> vector embedding.

NOTE ON THIS MVP: this environment cannot reach an embedding-model host
(e.g. Hugging Face) over the network, so this module uses TF-IDF
(scikit-learn) as a local, dependency-light stand-in for a real
sentence-embedding model. It's a legitimate lexical retrieval baseline,
not a semantic embedding — it will miss paraphrases/synonyms that a
real embedding model would catch. The interface below (`fit`,
`transform`, `similarity`) is intentionally the same shape a real
embedding client would expose, so swapping in a hosted embedding model
(and Qdrant for storage) later is a drop-in replacement, not a rewrite.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextEmbedder:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TextEmbedder must be fit() before transform()")
        return self._vectorizer.transform(texts).toarray()

    def similarity(self, query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
        """Cosine similarity of a single query vector against a matrix
        of corpus vectors. Returns a 1D array of scores."""
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        return cosine_similarity(query_vec, corpus_vecs)[0]
