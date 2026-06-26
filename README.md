# Production RAG Chatbot

A production-oriented Retrieval-Augmented Generation (RAG) chatbot built using **Streamlit**, **LangChain**, **ChromaDB**, **BM25**, **Cross-Encoder Reranking**, **Ollama**, and **SQLite**.

The system supports multiple isolated chat sessions, document-grounded responses, hybrid retrieval, query understanding, metadata filtering, streaming responses, and an evaluation framework for measuring retrieval quality.

---

# Features

## Retrieval

- Hybrid Search (Dense + Sparse)
- ChromaDB Vector Search
- BM25 Sparse Retrieval
- Ensemble Retrieval
- Cross-Encoder Reranking (BAAI/bge-reranker-base)
- Metadata Filtering (Page-based)
- Duplicate Chunk Removal

## Query Understanding

- Conversation-aware Query Rewriting
- Query Classification
- Query Understanding Pipeline
- Metadata Extraction
- Optional Query Decomposition for complex questions

## Generation

- Streaming Responses
- Source Citations
- Local LLM Inference (Ollama)
- Cloud Model Support (Gemini/OpenAI)

## Persistence

- Multi-chat Support
- Chat-level Document Isolation
- SQLite Chat History
- Persistent Chroma Collections

## Evaluation

- Hit@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Faithfulness (LLM-as-Judge)
- Answer Relevance (LLM-as-Judge)
- Retrieval / Generation Latency Logging

---

# Architecture

```text
                     +----------------------+
                     |     Uploaded PDFs    |
                     +----------+-----------+
                                |
                                v
                    PyMuPDF4LLM Markdown Loader
                                |
                                v
                     Markdown Header Splitter
                                |
                                v
                 Recursive Character Splitter
                                |
                                v
                    MiniLM Embedding Generation
                                |
                                v
                        ChromaDB Vector Store
                                |
                                |
          +---------------------+----------------------+
          |                                            |
          v                                            v
    Dense Retriever                             BM25 Retriever
          |                                            |
          +---------------------+----------------------+
                                |
                         Ensemble Retriever
                                |
                                v
                    Cross-Encoder Reranker
                                |
                                v
                     Metadata Filtering
                                |
                                v
                    Context Formatting
                                |
                                v
                     Prompt Construction
                                |
                                v
                               LLM
                                |
                                v
                   Streaming Answer + Sources
```

---

# Tech Stack

## Frontend

- Streamlit

## LLM

- Ollama
- Llama 3.2
- Gemini (optional)
- OpenAI (optional)

## Retrieval

- LangChain
- ChromaDB
- BM25Retriever
- EnsembleRetriever
- BAAI/bge-reranker-base

## Embeddings

- sentence-transformers/all-MiniLM-L6-v2

## Database

- SQLite

## Document Processing

- PyMuPDF4LLM
- PyMuPDF

---

# Project Structure

```
RAG_ChatBot/
│
├── app.py
├── rag_chain.py
├── ingest.py
├── database.py
├── config.py
│
├── retrievers/
│   ├── vector.py
│   ├── bm25.py
│   ├── hybrid.py
│   ├── metadata.py
│   └── deduplicate.py
│
├── reranker/
│   └── cross_encoder.py
│
├── query/
│   ├── understanding.py
│   ├── prompt.py
│   └── classifier.py
│
├──eval.py
├── chroma_db/
├── Docs/
├── database.db
└── README.md
```

---

# Installation

```bash
git clone https://github.com/Shanmuk7347/RAG-ChatBot
cd RAG_ChatBot
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Pull an Ollama model

```bash
ollama pull llama3.2
```

(Optional)

```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

Run

```bash
streamlit run app.py
```

---

# Usage

1. Create a new chat.
2. Upload one or more PDFs.
3. Ingest documents.
4. Ask questions.
5. View retrieved source chunks.
6. Switch models/providers if required.

---

# Evaluation

The project contains an evaluation pipeline for comparing retrieval strategies.

Metrics:

- Hit@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Faithfulness (LLM-as-Judge)
- Answer Relevance (LLM-as-Judge)
- Retrieval Latency
- Generation Latency

The evaluation framework was used to compare:

- Dense Retrieval
- Hybrid Retrieval
- Hybrid + Cross-Encoder Reranker
- Hybrid + Query Understanding

---

# Current Status

Implemented

- Hybrid Retrieval
- Cross-Encoder Reranking
- Query Rewriting
- Query Understanding
- Metadata Filtering
- Streaming Responses
- Source Attribution
- Evaluation Pipeline
- Multi-chat Persistence

Future Work

- Latency Improvements
- Multimodal Retrieval
- Table & Image Understanding
- Adaptive Query Decomposition
- FastAPI Backend
- Docker Deployment
- Authentication