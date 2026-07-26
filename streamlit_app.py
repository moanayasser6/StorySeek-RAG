import streamlit as st
import pandas as pd
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer

# -----------------------
# Page
# -----------------------

st.set_page_config(
    page_title="StorySeek",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StorySeek")
st.subheader("Searching Ancient Stories with Modern AI")

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
