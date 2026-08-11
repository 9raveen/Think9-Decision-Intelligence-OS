import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import resources
from styles import inject

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.agents.cross_brand_agent import _outcome_sentiment

st.set_page_config(page_title="Decision Explorer", page_icon="🔍", layout="wide")
inject()
st.title("🔍 Decision Explorer")
st.caption("Browse the decision archive directly — by supplier or by brand — without asking a question first.")

decisions, documents, _ = resources.get_corpus()
doc_by_id = {d.doc_id: d for d in documents}

view = st.radio("Browse by", ["Supplier", "Brand"], horizontal=True)

sentiment_pill = {
    "positive": ("🟢 Positive track record", "t9-pill-green"),
    "negative": ("🔴 Negative track record", "t9-pill-yellow"),  # reuse yellow (no red pill defined)
    "mixed": ("🟡 Mixed track record", "t9-pill-yellow"),
}

if view == "Supplier":
    suppliers = sorted({s for d in decisions for s in d.supplier_tags()})
    selected = st.selectbox("Select a supplier", suppliers)
    related = [d for d in decisions if selected in d.supplier_tags()]
else:
    brands = sorted({d.brand for d in decisions})
    selected = st.selectbox("Select a brand", brands)
    related = [d for d in decisions if d.brand == selected]

related = sorted(related, key=lambda d: d.date)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"### {selected}")
with col2:
    st.markdown(f'<p style="text-align:right;color:#9aa4b2;">{len(related)} related decisions</p>', unsafe_allow_html=True)

if view == "Supplier" and related:
    used_by = sorted({d.brand for d in related})
    st.markdown(f"**Used by:** {', '.join(used_by)}")
    lines_by_product = {}
    for d in related:
        if d.product_line:
            lines_by_product.setdefault(d.product_line, set()).add(d.brand)
    if lines_by_product:
        st.markdown("**Product lines:**")
        for pl, brands_using in lines_by_product.items():
            st.write(f"- {pl}: {', '.join(sorted(brands_using))}")

st.divider()

for d in related:
    sentiment = _outcome_sentiment(d)
    label, css_class = sentiment_pill.get(sentiment, ("⚪ Neutral", "t9-pill-gray"))

    badges = []
    if d.review_required:
        badges.append("🔒 review required")
    if d.scope == "portfolio_relevant":
        badges.append("🌐 portfolio-relevant")
    if d.product_line:
        badges.append(f"📦 {d.product_line}")
    badge_html = " · ".join(f'<span class="t9-badge">{b}</span>' for b in badges)

    with st.expander(f"{d.decision_id} — {d.brand} ({d.function}) — {d.date}"):
        st.markdown(f'<span class="t9-pill {css_class}">{label}</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="t9-decision-card">
              <p><b>Problem:</b> {d.problem}</p>
              <p><b>Options considered:</b> {', '.join(d.options_considered)}</p>
              <p><b>Decision made:</b> {d.decision_made}</p>
              <p><b>Reason:</b> {d.reason}</p>
              <p><b>Outcome:</b> {d.outcome}</p>
              {badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if d.evidence_doc_ids:
            st.markdown("**Evidence:**")
            for eid in d.evidence_doc_ids:
                doc = doc_by_id.get(eid)
                if doc:
                    st.markdown(f"- *{doc.type}* ({doc.date}): {doc.content}")

if not related:
    st.info("No decisions found for this selection.")