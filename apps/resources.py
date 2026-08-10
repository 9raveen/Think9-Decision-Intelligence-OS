"""Shared, cached resources for the Streamlit app.

Both pages import this module. Loading the corpus and building the
retrieval pipeline is expensive enough (TF-IDF fit) that it should
happen once per session, not on every query — st.cache_resource handles
that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.graph.pipeline import run_query as _run_query
from src.ingestion.embed import TextEmbedder
from src.ingestion.loader import load_corpus
from src.retrieval.decision_retriever import DecisionRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import DocumentVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_resource(show_spinner="Loading Think9 decision archive...")
def get_pipeline():
    decisions, documents, queries = load_corpus(DATA_DIR)
    decision_retriever = DecisionRetriever(embedder=TextEmbedder())
    decision_retriever.index(decisions)
    document_store = DocumentVectorStore(embedder=TextEmbedder())
    document_store.upsert(documents)
    hybrid = HybridRetriever(decision_retriever, document_store)
    return hybrid, decisions, documents, queries


def get_corpus():
    _, decisions, documents, queries = get_pipeline()
    return decisions, documents, queries


def ask(query_text: str):
    hybrid, *_ = get_pipeline()
    return _run_query(hybrid, query_text)