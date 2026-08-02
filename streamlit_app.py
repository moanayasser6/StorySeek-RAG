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

# -----------------------
# Global CSS (Arabic Manuscript / One Thousand and One Nights inspired)
# -----------------------

st.markdown("""
<style>

    :root {
        --bg-color: #F8F4EC;
        --card-color: #FFFFFF;
        --primary-color: #7A5230;
        --accent-color: #C89B3C;
        --text-color: #2B2B2B;
        --border-color: #E8DDC7;
    }

    /* App background */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    /* Hide default streamlit chrome for a cleaner product feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    /* -----------------------
       Hero Section
    ------------------------ */
    .hero-wrapper {
        text-align: center;
        padding: 2.5rem 1.5rem 2rem 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }

    .hero-eyebrow {
        display: inline-block;
        font-size: 0.75rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent-color);
        font-weight: 600;
        margin-bottom: 0.75rem;
        border: 1px solid var(--accent-color);
        border-radius: 999px;
        padding: 0.25rem 0.9rem;
    }

    .title {
        font-size: 3rem;
        font-weight: 800;
        color: var(--primary-color);
        margin: 0.2rem 0 0.4rem 0;
        letter-spacing: 0.01em;
        font-family: "Georgia", "Amiri", serif;
    }

    .subtitle {
        font-size: 1.15rem;
        color: var(--text-color);
        opacity: 0.75;
        margin-bottom: 0.9rem;
        font-weight: 500;
    }

    .hero-description {
        max-width: 620px;
        margin: 0 auto;
        color: var(--text-color);
        opacity: 0.65;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* -----------------------
       Search Card
    ------------------------ */
    .search-card-label {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.3rem;
    }

    div[data-testid="stTextInput"] {
        border-radius: 14px;
    }

    div[data-testid="stTextInput"] input {
        background-color: var(--bg-color);
        border: 1.5px solid var(--border-color);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 1.05rem;
        color: var(--text-color);
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(200, 155, 60, 0.18);
    }

    /* Search button */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, var(--primary-color), #8f6540);
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.2rem;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 6px 16px rgba(122, 82, 48, 0.25);
        transition: all 0.15s ease-in-out;
        width: 100%;
    }

    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #8f6540, var(--accent-color));
        box-shadow: 0 8px 20px rgba(200, 155, 60, 0.35);
        transform: translateY(-1px);
    }

    /* -----------------------
       Generic card container
    ------------------------ */
    .manuscript-card {
        background-color: var(--card-color);
        border: 1px solid var(--border-color);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 4px 18px rgba(122, 82, 48, 0.08);
        margin-bottom: 1.6rem;
    }

    /* -----------------------
       Status pills (loaded artifacts)
    ------------------------ */
    .status-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 0.6rem;
    }

    .status-pill {
        background-color: #FBF7EE;
        border: 1px solid var(--border-color);
        color: var(--primary-color);
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* -----------------------
       Results
    ------------------------ */
    .results-heading {
        color: var(--primary-color);
        font-weight: 800;
        font-size: 1.4rem;
        margin: 1.6rem 0 1rem 0;
        border-bottom: 2px solid var(--accent-color);
        display: inline-block;
        padding-bottom: 0.25rem;
    }

    .result-card {
        background-color: var(--card-color);
        border: 1px solid var(--border-color);
        border-left: 5px solid var(--accent-color);
        border-radius: 16px;
        padding: 1.4rem 1.7rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(122, 82, 48, 0.08);
        transition: box-shadow 0.15s ease-in-out;
    }

    .result-card:hover {
        box-shadow: 0 8px 22px rgba(122, 82, 48, 0.15);
    }

    .result-card h4 {
        color: var(--primary-color);
        margin: 0 0 0.6rem 0;
        font-size: 1.15rem;
    }

    .result-card p {
        color: var(--text-color);
        line-height: 1.85;
        font-size: 1.02rem;
        margin: 0;
        direction: rtl;
        text-align: right;
    }

    .score-badge {
        display: inline-block;
        background-color: #FBF7EE;
        border: 1px solid var(--accent-color);
        color: var(--primary-color);
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        margin-bottom: 0.9rem;
    }

    /* -----------------------
       Sidebar
    ------------------------ */
    section[data-testid="stSidebar"] {
        background-color: #FBF7EE;
        border-right: 1px solid var(--border-color);
    }

    .sidebar-title {
        color: var(--primary-color);
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .sidebar-subtitle {
        color: var(--text-color);
        opacity: 0.6;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }

    .sidebar-section-title {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.1rem 0 0.5rem 0;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.3rem;
    }

    .sidebar-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        font-size: 0.88rem;
        color: var(--text-color);
    }

    .sidebar-item-label {
        opacity: 0.65;
    }

    .sidebar-item-value {
        font-weight: 700;
        color: var(--primary-color);
        background-color: #F2E9D8;
        padding: 0.15rem 0.6rem;
        border-radius: 8px;
        font-size: 0.8rem;
    }

    /* -----------------------
       Footer
    ------------------------ */
    .app-footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-color);
        color: var(--text-color);
        opacity: 0.55;
        font-size: 0.85rem;
        line-height: 1.7;
    }

    .app-footer .stack {
        color: var(--primary-color);
        font-weight: 600;
        opacity: 0.9;
    }

    hr {
        border-color: var(--border-color) !important;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:
    st.markdown('<div class="sidebar-title">📜 StorySeek</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Project Information</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-item">
        <span class="sidebar-item-label">Stories</span>
        <span class="sidebar-item-value">1002 Arabic Stories</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Retrieval</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-item">
        <span class="sidebar-item-label">Strategy</span>
        <span class="sidebar-item-value">Hybrid Search</span>
    </div>
    <div class="sidebar-item">
        <span class="sidebar-item-label">Vector Store</span>
        <span class="sidebar-item-value">FAISS</span>
    </div>
    <div class="sidebar-item">
        <span class="sidebar-item-label">Sparse Retrieval</span>
        <span class="sidebar-item-value">BM25</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-item">
        <span class="sidebar-item-label">Embeddings</span>
        <span class="sidebar-item-value">Sentence Transformers</span>
    </div>
    <div class="sidebar-item">
        <span class="sidebar-item-label">LLM</span>
        <span class="sidebar-item-value">Qwen</span>
    </div>
    <div class="sidebar-item">
        <span class="sidebar-item-label">Language</span>
        <span class="sidebar-item-value">Arabic</span>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Hero Section
# -----------------------

st.markdown("""
<div class="hero-wrapper">
    <div class="hero-eyebrow">Ancient Tales • Modern AI</div>
    <div class="title">📜 StorySeek</div>
    <div class="subtitle">Searching Ancient Stories with Modern AI</div>
    <div class="hero-description">
        A hybrid retrieval system blending dense semantic search and classical lexical matching
        to surface the most relevant passages from a rich collection of Arabic stories —
        in the spirit of the timeless One Thousand and One Nights.
    </div>
</div>
""", unsafe_allow_html=True)

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

st.markdown(f"""
<div style="text-align:center;">
    <span class="status-pill">✅ All artifacts loaded</span>
</div>
<div class="status-row">
    <span class="status-pill">📚 Chunks: {len(chunks_df)}</span>
    <span class="status-pill">🧠 Embedding Shape: {embeddings.shape}</span>
    <span class="status-pill">🔎 FAISS Vectors: {faiss_index.ntotal}</span>
</div>
""", unsafe_allow_html=True)

st.write("")

def normalize_scores(scores):
    """
    Normalizes a list of scores to a 0-1 range using min-max normalization.

    Parameters:
        scores (list): list of raw scores

    Returns:
        list: normalized scores between 0 and 1
    """
    scores = np.array(scores, dtype=float)

    # Avoid division by zero if all scores are the same
    if scores.max() == scores.min():
        return [1.0 for _ in scores]

    normalized = (scores - scores.min()) / (scores.max() - scores.min())
    return normalized.tolist()

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

    # -------- Normalize scores before fusion --------
    norm_faiss_scores = normalize_scores(faiss_scores[0])
    norm_bm25_scores = normalize_scores([bm25_scores[idx] for idx in bm25_indices])

    # -------- Hybrid Fusion --------
    hybrid_scores = {}

    for score, idx in zip(norm_faiss_scores, faiss_indices[0]):
        hybrid_scores[idx] = alpha * float(score)

    for score, idx in zip(norm_bm25_scores, bm25_indices):
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + (1 - alpha) * float(score)

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

# -----------------------
# Search Card
# -----------------------

st.markdown('<div class="manuscript-card">', unsafe_allow_html=True)

st.markdown('<div class="search-card-label">🔍 Ask about a story</div>', unsafe_allow_html=True)

query = st.text_input(
    "Enter your question",
    placeholder="مثال: قصة عن الشجاعة",
    label_visibility="collapsed"
)

st.write("")

search_clicked = st.button("✨ Search the Manuscripts", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Results
# -----------------------

if search_clicked:

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        results = hybrid_search(query)

        st.markdown('<div class="results-heading">📖 Retrieved Context</div>', unsafe_allow_html=True)

        for i, item in enumerate(results, start=1):

            st.markdown(f"""
<div class="result-card">
<h4>📖 Result {i}</h4>
<div class="score-badge">Similarity Score: {item['score']:.4f}</div>
<p>{item["chunk"]}</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------

st.markdown("""
<div class="app-footer">
    Built with<br/>
    <span class="stack">Python • Streamlit • BM25 • FAISS • Sentence Transformers • Qwen</span>
    <br/><br/>
    © StorySeek
</div>
""", unsafe_allow_html=True)
