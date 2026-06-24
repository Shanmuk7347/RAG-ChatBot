from retrievers.bm25 import get_bm25_retriever
from langchain_classic.retrievers import EnsembleRetriever
from retrievers.vector import get_vector_retriever

def get_hybrid_retriever(chat_id: str):
    vector = get_vector_retriever(chat_id)
    bm25 = get_bm25_retriever(chat_id)

    hybrid = EnsembleRetriever(
        retrievers=[vector, bm25],
        weights=[0.5, 0.5])

#checking with equal, weights

    return hybrid