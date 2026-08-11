# 30-Day Roadmap

Every item traces to a specific, root-caused gap in `ARCHITECTURE.md`.

## Week 1 — Retrieval & Reasoning Reliability

- **Fix the Q7 conflict trigger** in `cross_brand_agent.py`: replace the strict `{"positive","negative"}` set-equality check with a principled rule based on the decision's own structured outcome/review signals where available, with keyword-sentiment as a documented fallback only — validated against real decision records via unit test, not just the end-to-end Q7 label, so the fix generalizes rather than overfits the benchmark.
- **Re-validate the no-precedent relevance threshold** (Q4) against a larger, more representative corpus.
- **Make the Router agent actually route:** wire `RoutingResult.function` into `hybrid_retriever.py` as a filter/re-rank signal before results reach `cross_brand_agent`, using metadata already captured (brand, function, supplier, product line, tags). Currently the router only labels results for display after retrieval has already happened.
- **Add explicit decision-context to queries** (current brand, function, supplier/product/category) so the system can distinguish a brand's own historical decision from a true cross-brand precedent.
- Expand the evaluation set: paraphrased supplier query, irrelevant supplier query, same-brand historical query, conflicting-evidence query, legal high-stakes query, insufficient-evidence query.
- Re-run the full benchmark and report honestly, including any new failures surfaced.

## Week 2 — Real Data & Institutional Memory

- Build ingestion pipeline for Think9's actual decision records (structured logs, decks, Slack/email threads — format TBD with Think9).
- Normalize structured decisions and unstructured evidence into the existing schema, preserving provenance and timestamps.
- Validate real corpus quality against the same `validator.py` checks used for the synthetic MVP corpus.
- Reassess TF-IDF vs. LSA (and evaluate a real sentence-transformer/hosted embedding API, now unblocked outside the sandbox) at real corpus scale — the current calibration finding may not hold as corpus size and language variability grow.

## Week 3 — Governance, Access & Auditability

- Brand-level and role-level access control.
- Expand governance rules beyond the current 3 triggers, based on Think9's actual review-required categories.
- Full audit trail per query: query text, user/role, evidence retrieved, answer given, governance decision, timestamp.

## Week 4 — Pilot & Production Hardening

- Pilot with a small subset of brands before full 30+ brand rollout.
- Incremental indexing/caching so new decisions don't require a full corpus re-index.
- Monitor in production: retrieval recall, false-positive rate, review-trigger frequency, low-evidence query rate.
- Collect real user feedback from the pilot.
- Move off Streamlit to a production frontend integrated with Think9's internal tools — only after pilot validation.

This roadmap does not assume or claim access to Think9's actual internal systems; Week 2 onward depends on Think9 providing real decision data and confirming integration points.
