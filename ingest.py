from langchain_core.document_loaders import BaseLoader
from pymupdf4llm import to_markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import settings
from logger import logger
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import json

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10),
    before_sleep=lambda k: logger.warning(f"Retrying for {k.fn.__name__} ({k.attempt_number}/3) because {type(k.outcome.exception()).__name__}"), reraise=True)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10),
    before_sleep=lambda k: logger.warning(f"Retrying for {k.fn.__name__} ({k.attempt_number}/3) because {type(k.outcome.exception()).__name__}"), reraise=True)
def save_to_chroma(chunks: list[Document], embeddings: HuggingFaceEmbeddings, chat_id: str):
    Chroma.from_documents(chunks, embeddings, collection_name=get_collection_name(chat_id), persist_directory=settings.chroma_dir)

def save_chunks(chunks: list[Document], chat_id:str):
    save_dir = settings.chunks_dir
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{chat_id}.json")
    chunk_data = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        } for chunk in chunks]
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)

    else:
        existing_chunks = []

    chunk_data.extend(existing_chunks)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            chunk_data,
            f,
            ensure_ascii=False,
            indent=2
        )
    
    logger.info(f"Saved chunks to {settings.chroma_dir}/{chat_id}.json")
        

def build_vector_data(file_paths:list[str], chat_id:str):

    """ Loading Documents and storing them as vectors in ChromaDB """
    start = time.perf_counter()
    logger.info("Loading documents...")
    #Converting the text into markdown for structural Splitting
    documents = []
    for path in file_paths:
        loader = PyMuPDFMarkdownLoader(path)
        documents.extend(loader.load())

    if not documents:
        logger.error("Error: No PDFs found in the Docs directory. Halting ingestion.")
        return

    logger.info("Starting Markdown splitting...")
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
    logger.info("Splitting markdown into chunks for token limiting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = text_splitter.split_documents(md_splits)
    save_chunks(chunks, chat_id)

    logger.info("Creating embeddings...")
    #Using all-MiniLM-L6-v2 for embedding generation, it converts text in a 384-dimensional vector.
    embeddings = get_embeddings()

    logger.info("Generating Vectors and storing them on disk...")
    save_to_chroma(chunks, embeddings, chat_id)
    stop = time.perf_counter()
    logger.success("Database built and stored on disk.")
    logger.info(f"Loaded {len(file_paths)} documents")
    logger.info(f"Ingestion completed in {(stop - start):.2f}s ")

if __name__ == "__main__":
    build_vector_data([r"Docs\Attention is all you need.pdf", r"Docs\Benchmarking.pdf", r"Docs\EnlightenGAN_Deep_Light_Enhancement_Without_Paired_Supervision_compressed.pdf"], "test")
    