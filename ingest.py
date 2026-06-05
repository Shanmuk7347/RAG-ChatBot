from langchain_core.document_loaders import BaseLoader
from pymupdf4llm import to_markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class PyMuPDFMarkdownLoader(BaseLoader):
    def __init__(self, file_path:str):
        self.file_path = file_path

    def load(self):
        md_text = to_markdown(self.file_path, page_chunks=True)
        docs = []
        for page in md_text:

            docs.append(
                Document(
                    page_content=page["text"],
                    metadata={
                        "source": self.file_path,
                        "page": page["metadata"]["page_number"]
                    }
                )
            )

        return docs

def get_collection_name(chat_id:str):
    return f"chat_{chat_id}"

def build_vector_data(file_paths:list[str], chat_id:str):

    """ Loading Documents and storing them as vectors in ChromaDB """

    print("Loading documents...")
    #Converting the text into markdown for structural Splitting
    documents = []
    for path in file_paths:
        loader = PyMuPDFMarkdownLoader(path)
        documents.extend(loader.load())

    if not documents:
        print("Error: No PDFs found in the Docs directory. Halting ingestion.")
        return

    print("Starting Markdown splitting...")
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
    #Spliting using markdown splitter and linking them to their sources 
    md_splits = []
    for doc in documents:
        splits = md_splitter.split_text(doc.page_content)
        #Linking the split to its source
        for split in splits:
            split.metadata.update(doc.metadata)
        md_splits.extend(splits)

    #Reduce the tokens so that model dosen't go beyond its context window
    print("Splitting markdown into chunks for token limiting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = text_splitter.split_documents(md_splits)
    
    print("Creating embeddings...")
    #Using all-MiniLM-L6-v2 for embedding generation, it converts text in a 384-dimensional vector.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Generating Vectors and storing them on disk...")
    Chroma.from_documents(chunks, embeddings, collection_name=get_collection_name(chat_id), persist_directory="./chroma_db")

    print("Database built and stored on disk.")

if __name__ == "__main__":
    pass
    