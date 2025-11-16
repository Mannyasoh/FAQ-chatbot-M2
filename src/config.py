import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class RAGConfig(BaseModel):
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model to use"
    )
    generation_model: str = Field(
        default="gpt-4o-mini", description="LLM model for answer generation"
    )
    chunk_size: int = Field(
        default=500, description="Size of text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=100, description="Overlap between chunks in characters"
    )
    min_chunk_size: int = Field(default=50, description="Minimum chunk size")
    vector_dimension: int = Field(default=1536, description="Dimension of embeddings")
    search_k: int = Field(default=5, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity for retrieval"
    )
    bm25_weight: float = Field(
        default=0.4, description="Weight for BM25 scores in hybrid search"
    )
    vector_weight: float = Field(
        default=0.6, description="Weight for vector scores in hybrid search"
    )
    search_type: str = Field(
        default="hybrid", description="Search type: 'hybrid', 'bm25', 'vector', 'auto'"
    )
    max_tokens: int = Field(default=8192, description="Maximum tokens for generation")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_chunks_per_query: int = Field(
        default=10, description="Maximum chunks per query"
    )
    max_cost_per_query: float = Field(
        default=1.0, description="Maximum cost per query in USD"
    )
    max_daily_cost: float = Field(default=50.0, description="Maximum daily cost in USD")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    index_dir: Path = Field(
        default=Path("data/index"), description="Index storage directory"
    )
    metrics_file: Path = Field(
        default=Path("metrics/rag_metrics.csv"), description="Metrics file"
    )

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            generation_model=os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            vector_dimension=int(os.getenv("VECTOR_DIMENSION", "1536")),
            search_k=int(os.getenv("SEARCH_K", "5")),
            max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
            max_chunks_per_query=int(os.getenv("MAX_CHUNKS", "10")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.4")),
            vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.6")),
            search_type=os.getenv("SEARCH_TYPE", "hybrid"),
        )

    def validate_config(self) -> bool:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if self.chunk_size < self.min_chunk_size:
            raise ValueError(f"chunk_size must be at least {self.min_chunk_size}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return True
