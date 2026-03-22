from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_pipeline import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_GENERATOR_MODEL,
    OPTIONAL_API_MODEL,
    answer_question,
    build_retriever_for_mode,
)

st.set_page_config(page_title="Chapter 3 RAG Chatbot", page_icon="📘", layout="wide")

st.title("📘 Chapter 3 QA Chatbot: Contextual Retrieval")
st.caption("Student ID: 125983 | Assigned chapter: 3 | Topic: N-gram Language Models")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k retrieved chunks", min_value=2, max_value=6, value=3, step=1)
    generator_mode = st.selectbox(
        "Generator mode",
        options=["extractive", "auto"],
        index=0,
        help="extractive = fully local and reproducible. auto = use gpt-4o-mini only if OPENAI_API_KEY is available; otherwise it falls back automatically.",
    )
    st.markdown("**Retriever model**")
    st.write(DEFAULT_EMBED_MODEL)
    st.markdown("**Default generator**")
    st.write(DEFAULT_GENERATOR_MODEL)
    st.markdown("**Optional API generator**")
    st.write(OPTIONAL_API_MODEL)
    st.info("The app always uses contextual retrieval in the backend, as required by the assignment.")

@st.cache_resource(show_spinner=True)
def load_pipeline():
    return build_retriever_for_mode(mode="contextual")

chunks, retriever = load_pipeline()

with st.form("qa_form"):
    st.subheader("Ask a question about Chapter 3")
    question = st.text_input(
        "Example: Why is lower perplexity better?",
        placeholder="Type your question here...",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question:
    with st.spinner("Retrieving relevant chunks and generating an answer..."):
        result = answer_question(
            query=question,
            chunks=chunks,
            retriever=retriever,
            top_k=top_k,
            generator=generator_mode,
        )

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown("### Source chunks used")
    for i, source in enumerate(result["sources"], start=1):
        label = f"Source {i} | {source['section']} | page {source['page']} | {source['chunk_id']} | score={source['score']}"
        with st.expander(label):
            st.caption(
                f"Citation: Chapter 3, {source['section']}, page {source['page']}, chunk {source['chunk_id']}"
            )
            st.write(source["text"])
elif submitted:
    st.warning("Please type a question first.")
else:
    st.write(
        "This chatbot is grounded in Chapter 3 and uses contextual retrieval so each chunk is prefixed "
        "with chapter, section, page, and short-topic context before retrieval."
    )
