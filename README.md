# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, ChromaDB, LangChain, Ollama, and SQLite.

The application allows users to upload PDF documents, create multiple isolated chat sessions, and ask questions grounded strictly in the uploaded documents.

---

## Features

* Multi-chat support
* Chat-level document isolation
* PDF ingestion and vectorization
* SQLite-based chat history persistence
* Source citation for every response
* Local LLM inference using Ollama

---

## Architecture

PDF Documents
->
PyMuPDF4LLM
->
Markdown Extraction
->
Markdown Header Splitter
->
Recursive Character Splitter
->
MiniLM Embeddings
->
ChromaDB Vector Store
->
Retriever
->
Llama 3.2 (Ollama)
->
Answer + Sources

---

## Tech Stack

### Frontend

* Streamlit

### LLM

* Ollama
* Llama 3.2

### RAG Framework

* LangChain

### Embeddings

* sentence-transformers/all-MiniLM-L6-v2

### Vector Database

* ChromaDB

### Persistence

* SQLite

### PDF Processing

* PyMuPDF4LLM
* PyMuPDF

---

## Project Structure

RAG_ChatBot/

├── app.py

├── rag_chain.py

├── ingest.py

├── database.py

├── chroma_db/

├── Docs/

├──database.db

└── .env


---

## Installation

### Clone Repository

```bash
git clone <https://github.com/Shanmuk7347/RAG-ChatBot>
cd RAG_ChatBot
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pull Llama 3.2/ qwen3/ gemma3

```bash
ollama pull llama3.2
```

### Save your API keys If you want to use Cloud models

```bash
GEMINI_API_KEY = "Your API KEY"
OPENAI_API_KEY = "Your API KEY"
```

### Run Application

```bash
streamlit run app.py
```

---

## Usage

1. Create a new chat.
2. Upload one or more PDF documents.
3. Click **Ingest and Vectorize**.
4. Ask questions about the uploaded documents.
5. View retrieved source chunks for every answer.
6. Create additional chats for isolated document collections.
7. Change provider and model for cloud models

---

## Current Capabilities

* Persistent chat history
* Persistent document tracking
* Chat-specific vector databases
* Local inference without external APIs
* Source-grounded responses

---

## Future Improvements

* Hybrid search (BM25 + Vector Search)
* Reranking pipeline
* Chat renaming
* Conversation memory
* Docker deployment
* FastAPI backend
* Authentication and user accounts

---

## Example Queries

* Explain the EM Algorithm from the uploaded notes.
* Summarize the key concepts in this chapter.
* What are the assumptions of K-Means clustering?
* Compare the methods discussed in the document.

---