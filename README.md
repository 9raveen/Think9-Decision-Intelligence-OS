# Think9 Decision Intelligence OS

**Has Think9 already learned this somewhere else?**

A centralized decision-intelligence layer for a 30+ brand consumer portfolio:
structured decision memory + cross-brand precedent discovery, not a
per-brand document chatbot.

Built for the Think9 Consumer AI & Data Science Intern take-home challenge.

## Status: Phase 1 — Data & Retrieval Foundation

Implemented:
- Pydantic schemas for Decisions, Documents, and Evaluation Queries
- Explicit corpus validation (referential integrity across decisions/documents/queries)
- Structured decision retrieval (Path A): filter by brand, function, supplier,
  product line, tags, date — plus semantic ranking over decision narrative text
- Document vector search (Path B): TF-IDF-based semantic search over
  unstructured evidence documents (see note below on the embedding approach)
- Hybrid retriever: combines both paths into a single evidence bundle.
  **Returns evidence only — no LLM reasoning, no precedent verdict.** That
  judgment belongs to the Cross-Brand Intelligence Agent (next phase).
- Evaluation harness against 7 ground-truth queries (3 cross-brand precedent
  cases, 3 no-precedent cases, 1 conflicting-precedent case)

Not yet implemented (next phases): Router / Cross-Brand / Synthesis agents,
LangGraph orchestration, deterministic governance rules, Streamlit UI (Ask
Think9 + Decision Explorer).

## Note on the embedding approach

This environment's network can't reach huggingface.co (verified: HTTP 403),
so a real sentence-transformer model can't be downloaded here. Two
embedding paths now exist side by side for comparison:

- `src/ingestion/embed.py` — **TF-IDF baseline**, sparse lexical matching.
- `src/ingestion/semantic_embed.py` — **LSA (TF-IDF + Truncated SVD)**, a
  genuinely different dense embedding, honestly labeled as *not*
  transformer-quality — see the module docstring for why and what it can't
  do. Backed by `src/retrieval/qdrant_store.py` (Qdrant in embedded/local
  mode, no server needed).

Run `python eval/run_comparison.py` for the full TF-IDF vs. semantic
comparison. **Current finding: LSA did not outperform TF-IDF on this
corpus** — Q1–Q3 decision/evidence recall were identical, and the
no-precedent false-positive rate got worse under the same threshold (see
comparison table in the eval output / project notes). This is a real,
reported result, not tuned to look better — see `docs/` for the
interpretation and recommended next step once written up.

Swapping in a real hosted embedding model later is a drop-in replacement of
these two files — no caller code changes required.

## Running it

```bash
pip install -r requirements.txt
python eval/run_eval.py    # runs Q1-Q7 against the retrieval layer
python -m pytest tests/    # unit tests
```

## Repository structure

```
data/               approved synthetic corpus (18 decisions, 28 documents, 7 eval queries)
src/schemas/         Pydantic models
src/ingestion/        loading, validation, embedding
src/retrieval/         decision retriever, document vector store, hybrid retriever
src/governance/        deterministic rules (stub — next phase)
src/agents/            router / cross-brand / synthesis agents (not yet built)
src/graph/              LangGraph pipeline (not yet built)
app/                     Streamlit UI (not yet built)
eval/                    metrics + eval runner
tests/                   unit tests
docs/                    architecture notes, roadmap, demo script
```

## Architecture

See `ARCHITECTURE.md` for the full data-flow diagram and the explicit
retrieval/reasoning boundary this project enforces throughout.
