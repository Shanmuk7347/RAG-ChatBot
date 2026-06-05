from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ingest import get_collection_name
import streamlit as st


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
def get_llm(model= "llama3.2"):
    return ChatOllama(model=model, temperature=0.3)

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
def get_chain(chat_id:str):
    retriever = get_retriever(chat_id)
    # Initialize the llm 
    llm = get_llm()

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
        answer = get_chain(chat_id="default").invoke(question)
        print("\n ======== Answer ======== ")
        print(answer)