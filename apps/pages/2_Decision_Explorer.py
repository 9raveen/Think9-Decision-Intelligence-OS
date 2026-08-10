import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import resources

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.agents.cross_brand_agent import _outcome_sentiment

st.set_page_config(page_title="Decision Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Decision Explorer")
st.caption("Browse the decision archive directly — by supplier or by brand — without asking a question first.")

decisions, documents, _ = resources.get_corpus()
doc_by_id = {d.doc_id: d for d in documents}

view = st.radio("Browse by", ["Supplier", "Brand"], horizontal=True)

sentiment_icon = {"positive": "🟢", "negative": "🔴", "mixed": "🟡"}

if view == "Supplier":
    suppliers = sorted({s for d in decisions for s in d.supplier_tags()})
    selected = st.selectbox("Select a supplier", suppliers)
    related = [d for d in decisions if selected in d.supplier_tags()]
else:
    brands = sorted({d.brand for d in decisions})
    selected = st.selectbox("Select a brand", brands)
    related = [d for d in decisions if d.brand == selected]

related = sorted(related, key=lambda d: d.date)

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

st.markdown(f"**Related decisions:** {len(related)}")

for d in related:
    sentiment = _outcome_sentiment(d)
    icon = sentiment_icon.get(sentiment, "⚪")
    header = f"{icon} {d.decision_id} — {d.brand} ({d.function}) — {d.date}"
    with st.expander(header):
        st.write(f"**Problem:** {d.problem}")
        st.write(f"**Options considered:** {', '.join(d.options_considered)}")
        st.write(f"**Decision made:** {d.decision_made}")
        st.write(f"**Reason:** {d.reason}")
        st.write(f"**Outcome:** {d.outcome}")
        badges = []
        if d.review_required:
            badges.append("🔒 review required")
        if d.scope == "portfolio_relevant":
            badges.append("🌐 portfolio-relevant")
        if d.product_line:
            badges.append(f"📦 {d.product_line}")
        if badges:
            st.caption(" · ".join(badges))

        if d.evidence_doc_ids:
            st.markdown("**Evidence:**")
            for eid in d.evidence_doc_ids:
                doc = doc_by_id.get(eid)
                if doc:
                    st.markdown(f"- *{doc.type}* ({doc.date}): {doc.content}")

if not related:
    st.info("No decisions found for this selection.")