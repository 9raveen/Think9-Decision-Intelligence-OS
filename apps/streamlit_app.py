import streamlit as st
from styles import inject

st.set_page_config(page_title="Think9 Decision Intelligence OS", page_icon="🧠", layout="wide")
inject()

st.markdown(
    """
    <div class="t9-hero">
      <h1>🧠 Think9 Decision Intelligence OS</h1>
      <p><i>Has Think9 already learned this somewhere else?</i></p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("#### What this is")
    st.write(
        "A centralized decision-intelligence layer for a 30+ brand consumer "
        "portfolio — structured decision memory and cross-brand precedent "
        "discovery, not a per-brand document chatbot."
    )
    st.markdown("#### Navigate")
    st.write("**💬 Ask Think9** — ask a question, get an evidence-backed answer with cross-brand precedent, conflict flags, and a human-review gate where relevant.")
    st.write("**🔍 Decision Explorer** — browse suppliers or brands to see their decision history directly, without asking a question first.")

with col2:
    st.markdown("#### Why not just a chatbot")
    st.markdown(
        "- Answers grounded in **structured decision records**, not raw retrieved text\n"
        "- Cross-brand precedent surfaced **explicitly**\n"
        "- Conflicting outcomes **flagged for human review**, never silently resolved\n"
        "- No fabricated confidence percentages"
    )

st.info(
    "This is an MVP prototype built on a synthetic corpus (18 decisions, "
    "28 supporting documents) — a stand-in for Think9's real 30+ brand scale. "
    "**Fictional brands:** Nova, Aura, Verve, Kindle, Lumen. "
    "**Fictional suppliers:** Alpha, Beta, Gamma. "
    "See the repository README and ARCHITECTURE.md for what's real MVP vs. roadmap."
)