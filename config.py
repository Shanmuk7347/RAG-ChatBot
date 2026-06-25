from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

class Settings(BaseSettings):
    #API Keys
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    #Paths
    db_path: str = "database.db"
    chroma_dir: str = "./chroma_db"
    docs_dir: str = "./Docs"
    log_dir: str = "./logs"
    chunks_dir: str = "./Data"

    #Embeddings model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    #Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 150

    #Retrival
    top_k: int = 10
    fetch_k:int = 20

    #Reranker
    reranker_model: str = "BAAI/bge-reranker-base"

    #Uploads
    upload_size_limit: int = 20
    allowed_extentions: tuple = (".pdf",)

    #Defaults
    default_provider: str = "Ollama"
    default_model: str = "llama3.2"

    class Config:
        env_file = ".env"
        populate_by_name = True

settings = Settings()