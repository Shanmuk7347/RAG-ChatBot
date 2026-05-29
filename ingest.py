from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_data():

    """ Loading Doucuments and storing them as vectors in ChromaDB """

    print("Loading documents...")
    dirloader = DirectoryLoader("./Docs", glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=True)
    documents = dirloader.load()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print("Creating embeddings...")
    #Using all-MiniLM-L6-v2 for embedding generation, it converts text in a 384-dimensional vector.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Generating Vectors and storing them on disk...")
    database = Chroma.from_documents(chunks, embeddings, collection_name="docs", persist_directory="./chroma_db")

    print("Database built and stored on disk.")

if __name__ == "__main__":
    build_vector_data()