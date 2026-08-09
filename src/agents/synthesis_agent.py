"""Decision Synthesis Agent.

Turns (EvidenceBundle, CrossBrandResult, GovernanceResult) into the final
answer text shown to the user. This is the one agent that calls an LLM —
everything upstream (retrieval, routing, conflict detection, governance)
is deterministic on purpose, so an LLM failure/hallucination here can't
corrupt what evidence was actually found.

Uses Groq (free tier, no card required) — same OpenAI-compatible
chat-completions shape. REQUIRES GROQ_API_KEY (see .env.example). Falls
back to a labeled template when no key is set — no interface change
needed either way. Get a free key at https://console.groq.com/
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.agents.cross_brand_agent import CrossBrandResult
from src.governance.rules import GovernanceResult
from src.retrieval.hybrid_retriever import EvidenceBundle

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the Think9 Decision Intelligence synthesis layer.
Given a query and evidence retrieved from Think9's decision archive, answer
the question: "has Think9 already learned this somewhere else?"

Rules:
- Ground every claim in the evidence provided. Cite decision IDs (e.g. D03)
  and document IDs (e.g. E05) for each claim.
- Do NOT invent a numeric confidence score. If asked for confidence, refer
  to the number and directness of corroborating records instead.
- If evidence shows a conflict (two brands with different outcomes for the
  same supplier/strategy), do not force a single verdict — present both
  sides and state that human review is recommended.
- If no relevant evidence was retrieved, say so plainly. Do not force a
  connection to loosely related evidence.
- Keep the answer concise: what precedent exists (or doesn't), what it
  shows, and what's recommended."""


@dataclass
class SynthesisResult:
    answer_text: str
    used_llm: bool


def _build_user_prompt(query_text, bundle, cross_brand, governance) -> str:
    lines = [f"Query: {query_text}", "", f"Cross-Brand Agent classification: {cross_brand.behavior}"]
    if cross_brand.relevant_decisions:
        lines.append("\nRelevant decisions:")
        for d in cross_brand.relevant_decisions:
            lines.append(f"- {d.decision_id} ({d.brand}, {d.function}): {d.problem} -> {d.outcome}")
    if cross_brand.conflicting_pairs:
        lines.append(f"\nConflicting pairs detected: {cross_brand.conflicting_pairs}")
    if governance.review_required:
        lines.append(f"\nGovernance: human review required. Reasons: {governance.reasons}")
    else:
        lines.append("\nGovernance: no human review flag triggered.")
    return "\n".join(lines)


def template_fallback(query_text, bundle, cross_brand, governance) -> str:
    parts = [f"[TEMPLATE FALLBACK — no GROQ_API_KEY configured] Query: {query_text}"]
    if cross_brand.behavior == "no_precedent_found":
        parts.append("No relevant precedent found in the decision archive for this query.")
    elif cross_brand.behavior == "conflict_flag_human_review":
        parts.append(
            f"Conflicting precedent found across {cross_brand.conflicting_pairs}. "
            "Outcomes differ across brands for related decisions — recommend human review "
            "rather than a single verdict."
        )
    else:
        ids = ", ".join(d.decision_id for d in cross_brand.relevant_decisions)
        parts.append(f"Relevant precedent found: {ids}.")
    if governance.review_required:
        parts.append(f"Human review required: {'; '.join(governance.reasons)}")
    return "\n".join(parts)


def synthesize(query_text: str, bundle: EvidenceBundle, cross_brand: CrossBrandResult, governance: GovernanceResult) -> SynthesisResult:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return SynthesisResult(answer_text=template_fallback(query_text, bundle, cross_brand, governance), used_llm=False)

    from groq import Groq

    client = Groq(api_key=api_key)
    user_prompt = _build_user_prompt(query_text, bundle, cross_brand, governance)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content
    return SynthesisResult(answer_text=text, used_llm=True)