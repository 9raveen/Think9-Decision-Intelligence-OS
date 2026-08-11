import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import resources
from styles import inject

import streamlit as st

st.set_page_config(page_title="Ask Think9", page_icon="💬", layout="wide")
inject()
st.title("💬 Ask Think9")
st.caption("Ask a question about a supplier, claim, or launch strategy. The answer is grounded in Think9's decision archive.")

decisions, documents, queries = resources.get_corpus()

with st.expander("Try an example question"):
    example_cols = st.columns(3)
    examples = [q.query_text for q in queries if q.type == "cross_brand_precedent"][:3]
    for col, ex in zip(example_cols, examples):
        if col.button(ex[:60] + ("..." if len(ex) > 60 else ""), use_container_width=True):
            st.session_state["query_input"] = ex

query = st.text_area(
    "Your question",
    key="query_input",
    placeholder="e.g. We're evaluating Supplier Alpha for packaging on a new SKU. What should we know?",
    height=80,
)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Searching decision archive and reasoning across brands..."):
        state = resources.ask(query)

    cross_brand = state["cross_brand"]
    governance = state["governance"]
    synthesis = state["synthesis"]
    routing = state["routing"]

    behavior_meta = {
        "surface_precedent_with_nuance": ("🟢 Relevant precedent found", "t9-pill-green"),
        "no_precedent_found": ("⚪ No relevant precedent found", "t9-pill-gray"),
        "conflict_flag_human_review": ("🟡 Conflicting precedent — human review recommended", "t9-pill-yellow"),
    }
    label, css_class = behavior_meta[cross_brand.behavior]
    st.markdown(f'<span class="t9-pill {css_class}">{label}</span>', unsafe_allow_html=True)

    if routing.function:
        st.caption(f"Routed as: **{routing.function}** — {routing.confidence_note}")

    st.markdown("#### Answer")

    # Clean rendering: parse the bracketed fallback instead of dumping it raw.
    answer_text = synthesis.answer_text
    if not synthesis.used_llm and answer_text.startswith("[TEMPLATE FALLBACK"):
        st.markdown('<span class="t9-fallback-badge">⚠️ Template fallback — set GROQ_API_KEY for a generated answer</span>', unsafe_allow_html=True)
        # Strip the bracket prefix and query echo, keep only the substantive part.
        body = answer_text.split("]", 1)[-1].strip()
        if "Query:" in body:
            body = body.split(".", 1)[-1].strip() if body.split(".", 1)[0].startswith("Query") else body
        st.markdown(f'<div class="t9-answer-card">{body}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="t9-answer-card">{answer_text}</div>', unsafe_allow_html=True)

    if cross_brand.relevant_decisions:
        st.markdown("#### Supporting decisions")
        for d in cross_brand.relevant_decisions:
            st.markdown(
                f"""
                <div class="t9-decision-card">
                  <span class="t9-decision-id">{d.decision_id}</span>
                  <span class="t9-decision-meta"> — {d.brand} ({d.function}, {d.date})</span>
                  <p><b>Problem:</b> {d.problem}</p>
                  <p><b>Decision:</b> {d.decision_made}</p>
                  <p><b>Outcome:</b> {d.outcome}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if cross_brand.conflicting_pairs:
        st.markdown("#### ⚠️ Conflicting precedent detected")
        for a, b in cross_brand.conflicting_pairs:
            st.write(f"- **{a}** and **{b}** disagree in outcome — see both records above before deciding.")

    if governance.review_required:
        st.markdown("#### 🔒 Governance")
        st.warning("Human review required.")
        for r in governance.reasons:
            st.write(f"- {r}")