import json
import os
import re
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from config import settings
from logger import logger
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import streamlit as st


def load_chunks(chat_id: str):
    """ 
    Load the chunks that are saved while ingesting,
    And make them as Document objects for BM25 retrieval
    """
    filepath = os.path.join(settings.chunks_dir, f"{chat_id}.json")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        logger.error(f"file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Chunk file is currupted: {filepath}")
        return []
    logger.info(f"Loaded {len(chunks)} for bm25 retrieval")
    return [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]

def preprocessing(text: str):
    """
    Preprocessing to remove redunctant words like the, a, etc.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = []
    for word in text.split():
        if word not in ENGLISH_STOP_WORDS:
            words.append(word)
    return words

def filter_docs(docs: list[Document], metadata_filters: dict | None):
    """
    Filter the documents based on the metadata filters
    """
    if metadata_filters is None:
        return docs

    filtered_docs = []
    for doc in docs:
        keys = metadata_filters.keys()
        match = True
        for key in keys:
            if doc.metadata.get(key) != metadata_filters.get(key):
                match = False
                break
        if match:
            filtered_docs.append(doc)

    return filtered_docs


@st.cache_resource(show_spinner=False)
def get_bm25_retriever(chat_id: str, k: int = settings.top_k, metadata_filters: dict | None = None):
    """ 
    Create a BM25 Retriever for a specific chat
    """
    docs = load_chunks(chat_id)
    filtered_docs = filter_docs(docs, metadata_filters)
    retriever = BM25Retriever.from_documents(filtered_docs, preprocess_func=preprocessing)
    retriever.k = k
    return retriever

if __name__ == "__main__": 
    bm25 = get_bm25_retriever("test", 10, {"page": 5, "source": r"Docs\Attention is all you need.pdf"})
    # "test" is test chat id
    # query is "What is scaled attention"
    print("After preprocessing query:", preprocessing("What is scaled attention"))
    answers = bm25.invoke("What is scaled attention")
    print(f"======Retrieved======")
    for doc in answers:
        print(f"Metadata: {doc.metadata}\npage_content: {doc.page_content[:100]}")
        print("-"*50)