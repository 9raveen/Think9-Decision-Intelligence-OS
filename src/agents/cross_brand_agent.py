"""Cross-Brand Intelligence Agent — classifies precedent-found /
no-precedent / conflict. Conflict heuristic is a documented, limited
keyword-sentiment approach — see docstring for exact limitation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

from src.retrieval.hybrid_retriever import EvidenceBundle
from src.schemas.decision import Decision

RELEVANCE_THRESHOLD = 0.12

NEGATIVE_OUTCOME_WORDS = {
    "delay", "delayed", "incident", "issue", "issues", "quarantine", "hold",
    "switch", "switched", "failed", "reject", "rejected", "recall",
    "underperform", "underperformed", "risk", "concern", "contamination",
    "destroyed", "defect", "defective", "violation", "noncompliant",
}
POSITIVE_OUTCOME_WORDS = {"reliable", "consistent", "resolved", "stable", "clean", "approved"}
NEGATION_TOKENS = {"no", "not", "without", "never"}
_WORD_RE = re.compile(r"[a-z']+")


def _outcome_sentiment(decision: Decision) -> str:
    text = (decision.outcome + " " + decision.reason).lower()
    positive_phrases = ["no issues", "no further", "met forecast", "sign-off", "no delay", "on schedule"]
    has_pos = any(p in text for p in positive_phrases)
    tokens = _WORD_RE.findall(text)
    has_neg = False
    for i, tok in enumerate(tokens):
        if tok in NEGATIVE_OUTCOME_WORDS:
            preceding = tokens[max(0, i - 4):i]
            if any(p in NEGATION_TOKENS for p in preceding):
                has_pos = True
            else:
                has_neg = True
        if tok in POSITIVE_OUTCOME_WORDS:
            has_pos = True
    if has_neg and not has_pos:
        return "negative"
    if has_pos and not has_neg:
        return "positive"
    return "mixed"


def _shares_entity(a: Decision, b: Decision) -> bool:
    if a.product_line or b.product_line:
        return bool(a.product_line and a.product_line == b.product_line)
    return bool(set(a.supplier_tags()) & set(b.supplier_tags()))


@dataclass
class CrossBrandResult:
    behavior: str
    relevant_decisions: list[Decision] = field(default_factory=list)
    conflicting_pairs: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""


def analyze(bundle: EvidenceBundle) -> CrossBrandResult:
    relevant = [sd.decision for sd in bundle.matched_decisions if sd.score >= RELEVANCE_THRESHOLD]
    if not relevant:
        return CrossBrandResult(behavior="no_precedent_found", note=f"No retrieved decision met the relevance threshold ({RELEVANCE_THRESHOLD}); treating as no relevant precedent.")

    conflicting_pairs: list[tuple[str, str]] = []
    for a, b in combinations(relevant, 2):
        if a.brand == b.brand:
            continue
        if _shares_entity(a, b):
            sa, sb = _outcome_sentiment(a), _outcome_sentiment(b)
            if {sa, sb} == {"positive", "negative"}:
                conflicting_pairs.append((a.decision_id, b.decision_id))

    if conflicting_pairs:
        return CrossBrandResult(behavior="conflict_flag_human_review", relevant_decisions=relevant, conflicting_pairs=conflicting_pairs, note=f"Outcome-language heuristic found {len(conflicting_pairs)} conflicting pair(s).")

    return CrossBrandResult(behavior="surface_precedent_with_nuance", relevant_decisions=relevant, note=f"{len(relevant)} relevant decision(s) found, no conflict detected.")