import json
import os
import re
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from config import settings
from logger import logger
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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

def get_bm25_retriever(chat_id: str, k: int = 4):
    """ 
    Create a BM25 Retriever for a specific chat
    """
    docs = load_chunks(chat_id)
    retriever = BM25Retriever.from_documents(docs, preprocess_func=preprocessing)
    retriever.k = k
    return retriever

if __name__ == "__main__": 
    bm25 = get_bm25_retriever("test", 3)
    # "test" is test chat id
    # query is "What is Schedule from day 10 - 20?"
    print("After preprocessing query:", preprocessing("What is Schedule from day 10 - 20?"))
    answers = bm25.invoke("What is Schedule from day 10 - 20?")
    print(f"======Retrieved======")
    for doc in answers:
        print(f"Metadata: {doc.metadata}\n page_content: {doc.page_content}")