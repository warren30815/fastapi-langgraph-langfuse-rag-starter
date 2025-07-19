from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str

    # LangFuse Configuration
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Database Configuration
    database_url: str = "postgresql://langfuse:langfuse@localhost:5432/langfuse"

    # Vector Database Configuration
    vector_db_path: str = "./data/vector_db"

    # Application Configuration
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # RAG Configuration
    chunk_size: int = 300
    chunk_overlap: int = 50
    max_tokens_per_chunk: int = 1000
    similarity_threshold: float = 0.5
    max_retrieval_results: int = 5

    # Agent Configuration
    max_iterations: int = 10
    temperature: float = 0.3
    max_tokens: int = 2000

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
