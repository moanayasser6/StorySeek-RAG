import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="StorySeek",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 StorySeek")
st.subheader("Searching Ancient Stories with Modern AI")

st.markdown(
    """
Search Arabic stories using a Hybrid Retrieval System
combining BM25, FAISS, and Large Language Models.
"""
)

st.divider()

# --------------------------------------------------
# Search Area
# --------------------------------------------------

query = st.text_input(
    "Enter your question:",
    placeholder="Example: قصة عن الشجاعة"
)

search_button = st.button("Search")

st.divider()

# --------------------------------------------------
# Results Area
# --------------------------------------------------

if search_button:

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:
        st.success("Search button works successfully!")

        st.write("Your query:")

        st.code(query)
           
