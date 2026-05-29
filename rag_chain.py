from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize embeddings same as in ingest.py
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load from local vector DB
data = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="docs"
)

# Create the retriever interface and get the top 4 relevant chunks using cosine similarity for a given query
retriever = data.as_retriever(search_kwargs={"k": 4})

# Initialize the llm 
llm = ChatOllama(model="llama3.2")

# Defining strict prompt architecture
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use only the context below to answer. If the answer is not in the context, say you don't know.\n\nContext: {context}"),
    ("human", "{question}")
])

# Construncting the Pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    print("Chatbot started. Type 'exit' to quit.")
    
    while True:
        question = input("\nAsk your PDF a question: ")
        if question.lower() in ["exit", "quit"]:
            break
            
        print("Thinking...")
        answer = chain.invoke(question)
        print("\n ======== Answer ======== ")
        print(answer)