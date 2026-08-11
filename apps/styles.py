"""Shared CSS injected across all pages."""
import streamlit as st

def inject():
    st.markdown(
        """
        <style>
        .stApp { background: #0e1117; }

        /* Hero */
        .t9-hero {
            padding: 2rem 2.2rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #1a1f2e 0%, #12161f 100%);
            border: 1px solid #2a3142;
            margin-bottom: 1.5rem;
        }
        .t9-hero h1 { margin: 0; font-size: 2rem; }
        .t9-hero p { color: #9aa4b2; font-size: 1.05rem; margin-top: 0.4rem; }

        /* Status pills */
        .t9-pill {
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            font-size: 0.82rem; font-weight: 600; margin-bottom: 0.6rem;
        }
        .t9-pill-green  { background: #10321f; color: #4ade80; border: 1px solid #1d5c37; }
        .t9-pill-yellow { background: #332705; color: #fbbf24; border: 1px solid #5c4a10; }
        .t9-pill-gray   { background: #1f2430; color: #9aa4b2; border: 1px solid #2a3142; }

        /* Answer card */
        .t9-answer-card {
            background: #161b26; border: 1px solid #2a3142; border-radius: 12px;
            padding: 1.2rem 1.4rem; margin: 0.6rem 0 1rem 0; line-height: 1.55;
        }
        .t9-fallback-badge {
            display: inline-block; font-size: 0.75rem; color: #fbbf24;
            background: #241e05; border: 1px solid #4a3c0a; border-radius: 6px;
            padding: 2px 8px; margin-bottom: 0.6rem;
        }

        /* Decision cards */
        .t9-decision-card {
            background: #161b26; border: 1px solid #2a3142; border-radius: 10px;
            padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
        }
        .t9-decision-id { color: #60a5fa; font-weight: 700; font-size: 0.9rem; }
        .t9-decision-meta { color: #9aa4b2; font-size: 0.82rem; }

        /* Badges row */
        .t9-badge { font-size: 0.78rem; color: #9aa4b2; margin-right: 0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )