"""Router: classifies which business function a query is about, from
retrieved decisions' own function field — no separate keyword list."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.retrieval.hybrid_retriever import EvidenceBundle

MIN_SCORE_FOR_ROUTING = 0.05


@dataclass
class RoutingResult:
    function: str | None
    confidence_note: str


def route(bundle: EvidenceBundle) -> RoutingResult:
    functions = [sd.decision.function for sd in bundle.matched_decisions if sd.score >= MIN_SCORE_FOR_ROUTING]
    if not functions:
        return RoutingResult(function=None, confidence_note="No retrieved decision scored above the routing threshold; treat as an unclassified/novel query.")
    counts = Counter(functions)
    top_function, top_count = counts.most_common(1)[0]
    return RoutingResult(function=top_function, confidence_note=f"{top_count}/{len(functions)} of the scored decisions above threshold are '{top_function}'.")