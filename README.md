# StorySeek-RAG
# StorySeek-RAG

### Intelligent Retrieval-Augmented Generation for Arabic Short Stories

StorySeek-RAG is an end-to-end Retrieval-Augmented Generation (RAG) system designed to intelligently search and answer questions about Arabic short stories.

The project combines **Hybrid Search (BM25 + FAISS)** with a **Large Language Model (Qwen 2.5 Instruct)** to retrieve relevant story passages and generate context-aware answers. The dataset consists of Arabic short stories with titles and story content, making the system capable of semantic search, question answering, and intelligent story retrieval.

---

# Project Overview

This project demonstrates the complete implementation of a Retrieval-Augmented Generation (RAG) pipeline, including:

* Data preprocessing
* Arabic text cleaning
* Story chunking
* Hybrid Retrieval (BM25 + FAISS)
* Embedding generation
* Ground Truth evaluation
* Large Language Model integration
* Streamlit-ready architecture

The notebook is organized as a complete end-to-end workflow suitable for research, learning, and portfolio purposes.

---

# Features

* Arabic Short Story Retrieval
* Hybrid Search (BM25 + FAISS)
* Semantic Search using Sentence Transformers
* Context-aware Question Answering
* Retrieval-Augmented Generation (RAG)
* Ground Truth Evaluation
* Streamlit-ready implementation
* Modular and reproducible workflow

---

# Dataset

Dataset:

**1002 Short Stories from Project Gutenberg**

Each record contains:

* Story Title
* Story Text

The dataset was preprocessed to create retrieval-ready text before chunk generation.

---

# Project Workflow

```text
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Create Retrieval Text
      │
      ▼
Chunk Stories
      │
      ▼
BM25 Index
      │
      ▼
Sentence Embeddings
      │
      ▼
FAISS Vector Index
      │
      ▼
Hybrid Search
      │
      ▼
Ground Truth Evaluation
      │
      ▼
Large Language Model
      │
      ▼
Generated Answer
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* JSON
* Sentence Transformers
* BM25 (rank-bm25)
* FAISS
* Hugging Face Transformers
* Qwen 2.5 Instruct
* Streamlit

---

# Repository Structure

```text
StorySeek-RAG/
│
├── Arabic_RAG_System.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── stories.json
│
├── outputs/
│   ├── chunks_df.csv
│   ├── embeddings.npy
│   ├── faiss_index.index
│   └── ground_truth.csv
│
└── images/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/StorySeek-RAG.git
cd StorySeek-RAG
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Install the required libraries.
3. Load the dataset.
4. Run all notebook cells sequentially.
5. Test the system using your own Arabic questions.

---

# Example Questions

* ما قصة الساعة التي توقفت؟
* احكِ لي قصة عن الشجاعة.
* ما الدرس المستفاد من هذه القصة؟
* ابحث عن قصة تتحدث عن الصداقة.
* من هو بطل القصة؟

---

# Future Improvements

* Deploy with Streamlit
* Support multiple Arabic datasets
* Improve retrieval ranking
* Add reranking models
* Support multilingual retrieval
* Optimize inference speed

---

# Acknowledgements

* Project Gutenberg
* Hugging Face
* Sentence Transformers
* FAISS
* Streamlit

---

# License

This project is intended for educational and portfolio purposes.

