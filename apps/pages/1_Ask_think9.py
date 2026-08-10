import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import resources

import streamlit as st

st.set_page_config(page_title="Ask Think9", page_icon="💬", layout="wide")
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

    behavior_labels = {
        "surface_precedent_with_nuance": ("🟢 Relevant precedent found", "success"),
        "no_precedent_found": ("⚪ No relevant precedent found", "info"),
        "conflict_flag_human_review": ("🟡 Conflicting precedent — human review recommended", "warning"),
    }
    label, kind = behavior_labels[cross_brand.behavior]
    getattr(st, kind)(label)

    if routing.function:
        st.caption(f"Routed as: **{routing.function}** — {routing.confidence_note}")

    st.markdown("#### Answer")
    if not synthesis.used_llm:
        st.caption("⚠️ Template fallback — set GROQ_API_KEY for a real generated answer (see .env.example).")
    st.write(synthesis.answer_text)

    if cross_brand.relevant_decisions:
        st.markdown("#### Supporting decisions")
        for d in cross_brand.relevant_decisions:
            with st.container(border=True):
                st.markdown(f"**{d.decision_id}** — {d.brand} ({d.function}, {d.date})")
                st.write(f"**Problem:** {d.problem}")
                st.write(f"**Decision:** {d.decision_made}")
                st.write(f"**Outcome:** {d.outcome}")

    if cross_brand.conflicting_pairs:
        st.markdown("#### ⚠️ Conflicting precedent detected")
        for a, b in cross_brand.conflicting_pairs:
            st.write(f"- **{a}** and **{b}** disagree in outcome — see both records above before deciding.")

    if governance.review_required:
        st.markdown("#### 🔒 Governance")
        st.warning("Human review required.")
        for r in governance.reasons:
            st.write(f"- {r}")