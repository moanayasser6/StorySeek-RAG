import streamlit as st
import pandas as pd
import numpy as np
import faiss
import joblib
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="StorySeek",
    page_icon="📚",
    layout="wide"
)

ARTIFACTS_PATH = "artifacts"

CHUNKS_PATH = f"{ARTIFACTS_PATH}/chunks.csv"
EMBEDDINGS_PATH = f"{ARTIFACTS_PATH}/embeddings.npy"
FAISS_PATH = f"{ARTIFACTS_PATH}/faiss_index.index"
BM25_PATH = f"{ARTIFACTS_PATH}/bm25.pkl"
