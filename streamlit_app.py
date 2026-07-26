import streamlit as st

st.set_page_config(
    page_title="StorySeek-RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StorySeek-RAG")

st.subheader("Intelligent Retrieval-Augmented Generation for Arabic Short Stories")

st.markdown(
    """
Welcome to **StorySeek-RAG**.

This application allows you to search Arabic short stories using
Hybrid Search (BM25 + FAISS) and generate answers using a Large Language Model.
"""
)

st.divider()

question = st.text_area(
    "Ask your question:",
    placeholder="مثال: احكي لي قصة عن الصداقة"
)

if st.button("🔍 Search"):
    st.info("The search pipeline will be connected in the next step.")
