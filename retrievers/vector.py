from langchain_chroma import Chroma
import streamlit as st
from config import settings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from logger import logger
from ingest import get_collection_name


@st.cache_resource(show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10),
    before_sleep=lambda k: logger.warning(f"Retrying for {k.fn.__name__} ({k.attempt_number}/3) because {type(k.outcome.exception()).__name__}"), reraise=True)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

def get_vectorstore(chat_id: str):
    return Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
        collection_name=get_collection_name(chat_id)
        )

def get_vector_retriever(chat_id: str):
    return get_vectorstore(chat_id).as_retriever(search_type= "mmr", search_kwargs= {"k": settings.top_k, "fetch_k": settings.fetch_k})