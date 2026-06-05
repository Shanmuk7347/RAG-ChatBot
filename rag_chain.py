from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from ingest import get_collection_name
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


Gemini_API_KEY = os.getenv("GEMINI_API_KEY")
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings same as in ingest.py
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load from local vector DB
@st.cache_resource(show_spinner=False)
def get_vectorstore(chat_id:str):

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_embeddings(),
        collection_name=get_collection_name(chat_id)
    )

@st.cache_resource(show_spinner=False)
def get_llm(provider="Ollama", model= "llama3.2"):
    if provider == "Ollama":
        return ChatOllama(model=model, temperature=0.3)
    elif provider == "Google":
        return ChatGoogleGenerativeAI(model=model, api_key=Gemini_API_KEY, streaming=True)
    elif provider == "OpenAI":
        return ChatOpenAI(model=model, temperature=0.3, api_key=OpenAI_API_KEY, streaming=True)


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
    llm = get_llm(provider, model)
    return (rewrite_prompt | llm | StrOutputParser())

Prompt = ChatPromptTemplate.from_messages([
        ("system", """Use only the provided context to answer.
        If the answer is not contained in the context,
        say:
        "I could not find that information in the uploaded documents."
        Be concise and accurate.{context}"""),
        ("human", "{question}")
    ])

def get_retriever(chat_id:str):
# Create the retriever and get the top 4 relevant chunks using mmr retrival
    return get_vectorstore(chat_id).as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

def get_chain(chat_id:str, provider, model):
    retriever = get_retriever(chat_id)

    # Initialize the llm 
    llm = get_llm(provider=provider, model=model)

    # Defining strict prompt architecture
    prompt = Prompt

    # Construncting the Pipeline
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt| llm| StrOutputParser()
    )

def delete_chat_collection(chat_id:str):
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
        answer = get_chain(chat_id="default", provider="Ollama", model="llama3.2").invoke(question)
        print("\n ======== Answer ======== ")
        print(answer)