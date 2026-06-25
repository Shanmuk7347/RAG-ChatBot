from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import time
from config import settings
from logger import logger
from retrievers.hybrid import get_hybrid_retriever
from retrievers.bm25 import get_bm25_retriever

logger.info(f"Loading CrossEncoder: {settings.reranker_model}")
model = CrossEncoder(model_name_or_path=settings.reranker_model)
logger.info("Reranker Loaded")

def reranker(query: str, docs: list[Document], top_k: int = 4, return_scores: bool = False):

    if not docs:
        return []
    
    pairs = [(query, doc.page_content) for doc in docs]
    start = time.perf_counter()
    scores = model.predict(pairs)
    elapsed = time.perf_counter() - start
    logger.info(f"Scored {len(docs)}-docs in {elapsed:.2f}s")
    doc_scores = list(zip(docs, scores))
    doc_scores.sort(key=lambda k: k[1], reverse=True)

    if return_scores:
        return doc_scores[:top_k]
    
    reranked_docs = [ doc for doc, _ in doc_scores[:top_k]]
    return reranked_docs

if __name__ == "__main__":
    query = "Why is a scaling factor applied in scaled dot-product attention?"
    hybrid = get_hybrid_retriever("test")
    docs = hybrid.invoke(query)

    reranked = reranker(query, docs, return_scores=True)

    for i, doc_score in enumerate(reranked, 1):
        print("="*50)
        print(f"{i}. Score: {doc_score[1]}")
        print(f"{i}. Page: {doc_score[0].metadata["page"]}")
        print(f"page_content: {doc_score[0].page_content}")