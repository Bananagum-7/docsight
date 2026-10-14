"""Streamlit UI. Run with: streamlit run app.py

Loads the pipeline in-process (no separate API needed), so it deploys as a
single Hugging Face Space.
"""

import streamlit as st

from docsight.pipeline import load_pipeline

st.set_page_config(page_title="DocSight", page_icon="📘", layout="centered")


@st.cache_resource(show_spinner="Loading index and models...")
def get_pipeline():
    return load_pipeline()


st.title("DocSight")
st.caption("Answers from the FastAPI documentation, with a citation for every claim. Says so when the docs don't cover it.")

examples = [
    "How do I run code after returning a response?",
    "How do I allow requests from a React app on another port?",
    "How do I close a database session after each request?",
    "Does FastAPI support GraphQL subscriptions natively?",
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state["q"] = ex

question = st.text_input("Ask a question", key="q")
if question:
    with st.spinner("Searching the docs..."):
        ans = get_pipeline().ask(question)
    if ans.abstained:
        st.warning(ans.answer)
        st.caption(f"Why: {ans.abstain_reason}")
    else:
        st.markdown(ans.answer)
        st.subheader("Sources")
        for c in ans.citations:
            st.markdown(f"**[{c.n}]** [{c.breadcrumb}]({c.url})")
    with st.expander("Debug: timings, tokens, verification"):
        st.json({"timings_ms": {k: round(v, 1) for k, v in ans.timings_ms.items()}, "usage": ans.usage, "verification": ans.verification})
