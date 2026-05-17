"""
Application configuration module.
Loads settings from environment variables or .env file.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    LLM_PROVIDER: str = Field(default="gemini", description="LLM provider: 'gemini', 'ollama', or 'huggingface'")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash", description="Gemini model name")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    OLLAMA_MODEL: str = Field(default="llama3", description="Ollama model name")

    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformers model for embeddings")

    # Vector Store Configuration
    CHROMA_MODE: str = Field(default="cloud", description="ChromaDB mode: 'local' or 'cloud'")
    CHROMA_API_KEY: str = Field(default="", description="ChromaDB cloud API key")
    CHROMA_TENANT: str = Field(default="", description="ChromaDB cloud tenant ID")
    CHROMA_DATABASE: str = Field(default="", description="ChromaDB cloud database name")
    VECTOR_STORE_DIR: str = Field(default="./vector_store", description="Directory for local ChromaDB persistence")
    COLLECTION_NAME: str = Field(default="healthcare_docs", description="ChromaDB collection name")

    # Document Configuration
    DATA_DIR: str = Field(default="./data", description="Directory containing healthcare documents")
    CHUNK_SIZE: int = Field(default=500, description="Text chunk size for splitting documents")
    CHUNK_OVERLAP: int = Field(default=100, description="Overlap between text chunks")

    # RAG Configuration
    TOP_K_RESULTS: int = Field(default=10, description="Number of relevant chunks to retrieve")
    CONFIDENCE_THRESHOLD_HIGH: float = Field(default=0.75, description="Threshold for high confidence")
    CONFIDENCE_THRESHOLD_MEDIUM: float = Field(default=0.50, description="Threshold for medium confidence")

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()
