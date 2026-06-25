from retrievers.bm25 import get_bm25_retriever
from langchain_classic.retrievers import EnsembleRetriever
from retrievers.vector import get_vector_retriever
from config import settings

def get_hybrid_retriever(chat_id: str, k: int = settings.top_k, metadata_filters: dict | None = None):
    vector = get_vector_retriever(chat_id=chat_id, k=k, metadata_filters=metadata_filters)
    bm25 = get_bm25_retriever(chat_id=chat_id, k=k, metadata_filters=metadata_filters)

    hybrid = EnsembleRetriever(
        retrievers=[vector, bm25],
        weights=[0.6, 0.4])

#checking with equal weights

    return hybrid