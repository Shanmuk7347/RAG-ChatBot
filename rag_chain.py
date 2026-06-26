from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import streamlit as st
from config import settings
from logger import logger
from retrievers.vector import get_vectorstore
from retrievers.hybrid import get_hybrid_retriever
from query.understanding import expand_query
from reranker.cross_encoder import reranker
from retry import safe_stream


Gemini_API_KEY = settings.gemini_api_key
OpenAI_API_KEY = settings.openai_api_key

@st.cache_resource(show_spinner=False)
def get_llm(provider=settings.default_provider, model= settings.default_model):
    logger.info(f"Initializing LLM, provider: {provider}, model: {model}")
    if provider == "Ollama":
        return ChatOllama(model=model, temperature=0.3)
    elif provider == "Google":
        return ChatGoogleGenerativeAI(model=model, api_key=Gemini_API_KEY, streaming=True)
    elif provider == "OpenAI":
        return ChatOpenAI(model=model, temperature=0.3, api_key=OpenAI_API_KEY, streaming=True)

"""
=============================================
Rewriting Prompt
=============================================
"""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
        """
        Rewrite the user's latest question into a standalone question.
        Use the conversation history only to resolve references like:
        - it
        - they
        - this
        - that
        Return ONLY the rewritten question.
        """),
    ("human",
        """
        Conversation History:
        {history}
        Latest Question:
        {question}
""")
])

def rewriter(provider, model):
    logger.info(f"rewriting prompt")
    llm = get_llm(provider, model)
    return (rewrite_prompt | llm | StrOutputParser())


def deduplicate(docs: list[Document]):
    """
    removes the duplicate chucks that are retrieved
    """
    seen = set()
    unique_docs = []

    for doc in docs:
        key = (doc.metadata.get("source"),
               doc.metadata.get("page"),
               doc.page_content,)
        
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs

def understand_and_rerank(chat_id: str, query: str):
    """
    Rewrite the query, give sub_queries and metadata-filtering and reranks them using CrossEncoder
    """
    logger.info("Understanding the query")
    understanding = expand_query(query)
    query_type = understanding["query_type"]
    rewritten_query = understanding["rewritten_query"]
    metadata_filter = understanding["metadata_filter"]
    sub_queries = understanding["sub_queries"]
    docs = None
    if query_type == "simple":
        docs = get_hybrid_retriever(chat_id, metadata_filters=metadata_filter).invoke(rewritten_query)

    elif query_type == "filtered":
        docs = get_hybrid_retriever(chat_id, metadata_filters=metadata_filter).invoke(rewritten_query)

    elif query_type == "multi":
        all_docs = []
        retriever =  get_hybrid_retriever(chat_id, metadata_filters=metadata_filter)
        for query in sub_queries:
            retrieved_docs = retriever.invoke(query)
            all_docs.extend(retrieved_docs)
        
        docs = deduplicate(all_docs)

    reranked_docs = reranker(rewritten_query, docs)

    return rewritten_query, reranked_docs
    
Prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a Retrieval-Augmented Generation (RAG) assistant.

        Answer the user's question ONLY using the retrieved context below.

        If the answer is present in the context, answer it directly.

        Ignore irrelevant chunks.

        If multiple chunks support the answer, combine them.

        If the answer is truly not contained in the context, reply exactly:

        "I could not find that information in the uploaded documents."

        ========================
        Retrieved Context
        ========================

        {context}
        """),
            ("human", "{question}")
        ])

from langchain_core.documents import Document

def format_context(docs):
    chunks = []

    for i, doc in enumerate(docs, 1):
        chunks.append(
        f"""
        ========== Chunk {i} ==========
        Source: {doc.metadata.get("source")}
        Page: {doc.metadata.get("page")}

        Content:
        {doc.page_content}
        """)

    return "\n".join(chunks)

def ask(chat_id:str, query: str, docs: list[Document] ,provider: str = settings.default_provider, model: str = settings.default_model):
    """
    Takes query as input and returns Response
    """

    context = format_context(docs)

    # Initialize the llm 
    llm = get_llm(provider=provider, model=model)

    # Defining strict prompt architecture
    prompt = Prompt

    logger.info(f"Initilaizing chain for chat: {chat_id}")
    
    generation_chain =( prompt| llm| StrOutputParser())
    stream = safe_stream(
        generation_chain,
        {
            "context": context,
            "question": query
        }
    )
    for chunk in stream:
        yield chunk

def delete_chat_collection(chat_id:str):
    logger.info(f"Deleting chat: {chat_id}")
    vs = get_vectorstore(chat_id)
    vs.delete_collection()
    get_vectorstore.clear()


if __name__ == "__main__":
    print("Chatbot started. Type 'exit' to quit.")
    
    while True:
        question = input("\nAsk your PDF a question: ")
        if question.lower() in ["exit", "quit"]:
            break
            
        print("Thinking...")
        answer = ask(chat_id="test", provider="Ollama", model="llama3.2", query=question)
        print("\n ======== Answer ======== ")
        print(answer)