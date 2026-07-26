import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="StorySeek-RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📚 StorySeek-RAG")

    st.markdown("---")

    st.markdown("### About")
    st.write(
        """
        StorySeek-RAG is an AI-powered system that retrieves
        Arabic short stories using Hybrid Search (BM25 + FAISS)
        and generates answers with a Large Language Model.
        """
    )

    st.markdown("---")

    st.markdown("### Technologies")

    st.markdown("""
- BM25
- FAISS
- Sentence Transformers
- Qwen LLM
- Streamlit
""")

# -----------------------------
# Main Page
# -----------------------------
st.title("📚 StorySeek-RAG")

st.subheader(
    "Searching Ancient Stories with Modern AI"
)

st.write(
    """
Ask a question about Arabic short stories.
The system will retrieve the most relevant story passages
and generate an AI-powered answer.
"""
)

st.divider()

# -----------------------------
# User Question
# -----------------------------
question = st.text_area(
    "Your Question",
    placeholder="مثال: احكي لي قصة عن الشجاعة",
    height=120
)

search = st.button(
    "🔍 Search",
    use_container_width=True
)

st.divider()

# -----------------------------
# Results Area
# -----------------------------
st.subheader("Retrieved Context")

context_placeholder = st.empty()

st.subheader("AI Answer")

answer_placeholder = st.empty()

# -----------------------------
# Temporary Behavior
# -----------------------------
if search:

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching stories..."):

            context_placeholder.info(
                "Hybrid Search will be connected in the next step."
            )

            answer_placeholder.success(
                "LLM response will appear here."
            )
