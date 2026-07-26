import streamlit as st
import pandas as pd
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
# -----------------------
# Page
# -----------------------

st.set_page_config(
    page_title="StorySeek",
    page_icon="📚",
    layout="wide"
)
st.markdown("""
<style>

/* هنا كود الـ CSS بالكامل */

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title">
📜 StorySeek
</div>

<div class="subtitle">
Searching Ancient Stories with Modern AI
</div>
""", unsafe_allow_html=True)
search = st.button("Search")
st.markdown(
'</div>',
unsafe_allow_html=True
)


# -----------------------
# Paths
# -----------------------

ARTIFACTS = "artifacts"

CHUNKS = f"{ARTIFACTS}/chunks.csv"
EMBEDDINGS = f"{ARTIFACTS}/embeddings.npy"
FAISS_INDEX = f"{ARTIFACTS}/faiss_index.index"
BM25 = f"{ARTIFACTS}/bm25.pkl"

# -----------------------
# Load Resources
# -----------------------

@st.cache_resource
def load_resources():

    chunks_df = pd.read_csv(CHUNKS)

    embeddings = np.load(EMBEDDINGS)

    faiss_index = faiss.read_index(FAISS_INDEX)

    bm25 = joblib.load(BM25)

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return chunks_df, embeddings, faiss_index, bm25, embedding_model

chunks_df, embeddings, faiss_index, bm25, embedding_model = load_resources()

st.success("✅ All project artifacts loaded successfully!")

st.write("Chunks:", len(chunks_df))
st.write("Embedding Shape:", embeddings.shape)
st.write("FAISS Vectors:", faiss_index.ntotal)

def hybrid_search(query, top_k=5, alpha=0.7):

    # -------- Dense Search (FAISS) --------
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    faiss_scores, faiss_indices = faiss_index.search(
        query_embedding.astype("float32"),
        top_k
    )

    # -------- Sparse Search (BM25) --------
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

    # -------- Hybrid Fusion --------
    hybrid_scores = {}

    for score, idx in zip(faiss_scores[0], faiss_indices[0]):
        hybrid_scores[idx] = alpha * float(score)

    for idx in bm25_indices:
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + (1 - alpha) * float(bm25_scores[idx])

    ranked = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked[:top_k]:

        results.append({
            "score": score,
            "chunk": chunks_df.iloc[idx]["chunk_text"]
        })

    return results

query = st.text_input(
    "Enter your question",
    placeholder="مثال: قصة عن الشجاعة"
)

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        results = hybrid_search(query)

        st.subheader("Retrieved Context")

        for i, item in enumerate(results, start=1):

            st.markdown(f"### Result {i}")

            st.write(f"Score: {item['score']:.4f}")

            st.markdown(f"""
<div class="result-card">

<h4>📖 Story Result</h4>

<p>{item["chunk"]}</p>

</div>
""", unsafe_allow_html=True)

            st.divider()
