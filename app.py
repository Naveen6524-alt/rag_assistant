"""Streamlit UI:  streamlit run app.py"""
import importlib.util
import os

import streamlit as st

from cli import build_retriever
from rag.chunking import load_corpus

st.set_page_config(page_title="RAG assistant", page_icon="🔎", layout="wide")
st.title("🔎 RAG assistant over the FastAPI tutorial docs")
st.caption("Retrieval you can inspect: every answer is backed by the passages shown below it.")

HAS_DENSE = importlib.util.find_spec("sentence_transformers") is not None
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.header("Retrieval settings")
    options = ["bm25"] + (["dense", "hybrid"] if HAS_DENSE else [])
    kind = st.selectbox("Retriever", options)
    if not HAS_DENSE:
        st.caption("Dense/hybrid hidden: install sentence-transformers (Python 3.11/3.12).")
    chunk_words = st.select_slider("Chunk size (words)", [80, 180, 350], value=180)
    heading = st.checkbox("Prepend heading path to indexed text", value=True)
    k = st.slider("Passages to retrieve", 1, 8, 4)
    generate = st.checkbox("Generate an answer with an LLM", value=False, disabled=not HAS_KEY)
    if not HAS_KEY:
        st.caption("Set ANTHROPIC_API_KEY to enable answer generation.")


@st.cache_resource(show_spinner="Building index...")
def get_index(kind, chunk_words, heading):
    chunks = load_corpus("data/docs", chunk_words, 30)
    return chunks, build_retriever(kind, chunks, heading)


chunks, retriever = get_index(kind, chunk_words, heading)
st.write(f"Index: **{len(chunks)}** chunks from the docs folder.")

question = st.text_input("Ask a question", placeholder="How do I run something after the response is sent?")
if question:
    hits = [(chunks[i], s) for i, s in retriever.search(question, k)]
    if generate:
        from rag.generate import answer
        with st.spinner("Generating..."):
            try:
                st.subheader("Answer")
                st.write(answer(question, [c for c, _ in hits]))
            except Exception as e:  # show the error instead of crashing the page
                st.error(f"Generation failed: {e}")
    st.subheader("Retrieved passages")
    for n, (c, score) in enumerate(hits, start=1):
        with st.expander(f"[{n}] {c.source} > {c.heading}   (score {score:.2f})", expanded=(n == 1)):
            st.write(c.text)
