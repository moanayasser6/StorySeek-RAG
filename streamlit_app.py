import streamlit as st

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="StorySeek",
    page_icon="📚",
    layout="wide"
)

# -------------------------------
# Header
# -------------------------------

st.title("📚 StorySeek")

st.markdown(
"""
### Searching Ancient Stories with Modern AI

Search Arabic stories using a Hybrid Retrieval-Augmented Generation (RAG) system
powered by BM25, FAISS, Sentence Transformers, and Qwen.
"""
)

st.divider()

# -------------------------------
# Search Box
# -------------------------------

query = st.text_input(
    "Enter your question",
    placeholder="مثال: قصة عن الشجاعة"
)

search = st.button("Search")

st.divider()

# -------------------------------
# Results
# -------------------------------

if search:

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:
        st.success("Search button works successfully!")

        st.write("Your query:")

        st.code(query)
