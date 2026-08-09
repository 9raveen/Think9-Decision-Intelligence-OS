"""Deterministic governance rules — DEFERRED.

This file is a structural placeholder for the current phase. Per the
approved phasing, governance logic is implemented alongside the agent
layer (Router / Cross-Brand / Synthesis agents), not during the
data-and-retrieval foundation phase.

When implemented, this module must remain rule-based (e.g. `if
decision.review_required or decision.function == "legal": require
human review`), not an LLM call — governance is a deterministic gate,
not a judgment call delegated to a model.
"""
