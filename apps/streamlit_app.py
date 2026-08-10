import streamlit as st

st.set_page_config(page_title="Think9 Decision Intelligence OS", page_icon="🧠", layout="wide")

st.title("🧠 Think9 Decision Intelligence OS")
st.markdown("### *Has Think9 already learned this somewhere else?*")

st.markdown(
    """
A centralized decision-intelligence layer for a 30+ brand consumer portfolio —
structured decision memory and cross-brand precedent discovery, not a
per-brand document chatbot.

**Use the sidebar to navigate:**

- **Ask Think9** — ask a question, get an evidence-backed answer with
  cross-brand precedent, conflict flags, and a human-review gate where relevant.
- **Decision Explorer** — browse suppliers or brands to see their decision
  history directly, without asking a question first.

---

**How this differs from a document chatbot:** answers are grounded in
structured decision records (not just retrieved text), cross-brand
precedent is surfaced explicitly, conflicting outcomes across brands are
flagged for human review rather than resolved into a single false-confident
verdict, and nothing is answered with a fabricated confidence percentage.
"""
)

st.info(
    "This is an MVP prototype built on a synthetic corpus (18 decisions, "
    "28 supporting documents across 5 fictional brands). See the repository "
    "README and ARCHITECTURE.md for what's real MVP vs. roadmap."
)