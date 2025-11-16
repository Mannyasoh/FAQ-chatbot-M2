import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
import tiktoken
from loguru import logger
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RAGConfig
from src.costs import estimate_embedding_cost

logger.add("logs/hybrid_search.log", rotation="10 MB", level="INFO", serialize=True)
logger.add(
    "logs/hybrid_search_error.log", rotation="10 MB", level="ERROR", serialize=True
)


class HybridSearchStore:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunks: List[Tuple[str, Dict[str, Any]]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: BM25Okapi = None
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_weight = getattr(config, "bm25_weight", 0.4)
        self.vector_weight = getattr(config, "vector_weight", 0.6)
        self._cache_file = Path("data/embedding_cache.pickle")
        self._embedding_cache = self._load_embedding_cache()
        self._cache_hits = 0
        self._cache_misses = 0
        self.index_metadata: Dict[str, Any] = {}

    def add_chunks(self, chunks: List[Tuple[str, Dict[str, Any]]]) -> float:
        logger.info(f"Building hybrid index for {len(chunks)} chunks")
        self.chunks = chunks
        texts = [chunk[0] for chunk in chunks]
        logger.info("Building BM25 index")
        self.tokenized_corpus = [self._tokenize_text(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("Generating embeddings")
        (embeddings, cost) = self._generate_embeddings(texts)
        self.embeddings = embeddings
        self._enhance_metadata()
        self.index_metadata = {
            "total_chunks": len(chunks),
            "embedding_model": self.config.embedding_model,
            "vector_dimension": embeddings.shape[1],
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "total_embedding_cost": cost,
        }
        logger.success(f"Hybrid index built. Cost: ${cost:.6f}")
        return cost

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
    ) -> Tuple[List[Tuple[str, Dict, float]], float]:
        if not self.chunks:
            return ([], 0.0)
        k = k or self.config.search_k
        (query_embedding, cost) = self._generate_single_embedding(query)
        if search_type == "auto":
            search_type = self._determine_search_strategy(query)
        if search_type == "bm25":
            results = self._bm25_search(query, k * 2, metadata_filter)
        elif search_type == "vector":
            results = self._vector_search(query_embedding, k * 2, metadata_filter)
        else:
            results = self._hybrid_search(
                query, query_embedding, k * 2, metadata_filter
            )
        results = self._apply_metadata_filter(results, metadata_filter)
        results = results[:k]
        return (results, cost)

    def _hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Dict, float]]:
        bm25_scores = self._get_bm25_scores(query)
        bm25_rankings = np.argsort(bm25_scores)[::-1]
        vector_similarities = self._get_vector_similarities(query_embedding)
        vector_rankings = np.argsort(vector_similarities)[::-1]
        rrf_scores: Dict[int, float] = {}
        for i, doc_idx in enumerate(bm25_rankings):
            rrf_scores[int(doc_idx)] = rrf_scores.get(
                int(doc_idx), 0
            ) + self.bm25_weight / (60 + i + 1)
        for i, doc_idx in enumerate(vector_rankings):
            rrf_scores[int(doc_idx)] = rrf_scores.get(
                int(doc_idx), 0
            ) + self.vector_weight / (60 + i + 1)
        sorted_indices = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )
        results = []
        for idx in sorted_indices[:k]:
            if idx < len(self.chunks):
                (chunk_text, metadata) = self.chunks[idx]
                combined_score = rrf_scores[idx]
                metadata = metadata.copy()
                metadata["bm25_score"] = float(bm25_scores[idx])
                metadata["vector_similarity"] = float(vector_similarities[idx])
                metadata["rrf_score"] = combined_score
                results.append((chunk_text, metadata, combined_score))
        return results

    def _bm25_search(
        self, query: str, k: int, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, Dict, float]]:
        scores = self._get_bm25_scores(query)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if idx < len(self.chunks) and scores[idx] > 0:
                (chunk_text, metadata) = self.chunks[idx]
                metadata = metadata.copy()
                metadata["bm25_score"] = float(scores[idx])
                results.append((chunk_text, metadata, float(scores[idx])))
        return results

    def _vector_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Dict, float]]:
        similarities = self._get_vector_similarities(query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_indices:
            if (
                idx < len(self.chunks)
                and similarities[idx] >= self.config.similarity_threshold
            ):
                (chunk_text, metadata) = self.chunks[idx]
                metadata = metadata.copy()
                metadata["vector_similarity"] = float(similarities[idx])
                results.append((chunk_text, metadata, float(similarities[idx])))
        return results

    def _get_bm25_scores(self, query: str) -> np.ndarray:
        if self.bm25 is None:
            return np.zeros(len(self.chunks))
        query_tokens = self._tokenize_text(query)
        return np.array(self.bm25.get_scores(query_tokens))

    def _get_vector_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        if self.embeddings is None:
            return np.zeros(len(self.chunks))
        return cosine_similarity([query_embedding], self.embeddings)[0]

    def _determine_search_strategy(self, query: str) -> str:
        query_lower = query.lower()
        keyword_indicators = [
            "policy",
            "form",
            "document",
            "number",
            "date",
            "time",
            "hours",
        ]
        if any((indicator in query_lower for indicator in keyword_indicators)):
            return "bm25"
        concept_indicators = ["how to", "what is", "explain", "meaning", "understand"]
        if any((indicator in query_lower for indicator in concept_indicators)):
            return "vector"
        return "hybrid"

    def _apply_metadata_filter(
        self,
        results: List[Tuple[str, Dict, float]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, Dict, float]]:
        if not metadata_filter:
            return results
        filtered_results = []
        for text, metadata, score in results:
            match = True
            for key, value in metadata_filter.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_results.append((text, metadata, score))
        return filtered_results

    def _enhance_metadata(self):
        for i, (text, metadata) in enumerate(self.chunks):
            metadata["text_length"] = len(text)
            metadata["word_count"] = len(text.split())
            metadata["has_question"] = "?" in text
            metadata["has_numbers"] = any((char.isdigit() for char in text))
            text_lower = text.lower()
            if any(
                (
                    word in text_lower
                    for word in ["vacation", "pto", "time off", "leave"]
                )
            ):
                metadata["topic_category"] = "time_off"
            elif any(
                (
                    word in text_lower
                    for word in ["benefit", "health", "insurance", "401k"]
                )
            ):
                metadata["topic_category"] = "benefits"
            elif any(
                (
                    word in text_lower
                    for word in ["payroll", "salary", "pay", "overtime"]
                )
            ):
                metadata["topic_category"] = "payroll"
            elif any(
                (word in text_lower for word in ["onboard", "new employee", "start"])
            ):
                metadata["topic_category"] = "onboarding"
            else:
                metadata["topic_category"] = "general"
            self.chunks[i] = (text, metadata)

    def _tokenize_text(self, text: str) -> List[str]:
        import re

        tokens = re.findall("\\b\\w+\\b", text.lower())
        return tokens

    def _generate_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, float]:
        total_tokens = sum((len(self.tokenizer.encode(text)) for text in texts))
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model, input=texts
            )
            embeddings = np.array([item.embedding for item in response.data])
            cost = estimate_embedding_cost(self.config.embedding_model, total_tokens)
            return (embeddings, cost)
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise openai.AuthenticationError(
                "Invalid OpenAI API key for embeddings"
            ) from e
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise openai.RateLimitError(
                "OpenAI rate limit exceeded for embeddings"
            ) from e
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise openai.APIError(f"OpenAI API error for embeddings: {e}") from e
        except OSError as e:
            logger.error(f"Network or system error during embedding generation: {e}")
            raise OSError(f"Network error during embedding generation: {e}") from e

    def _generate_single_embedding(self, text: str) -> Tuple[np.ndarray, float]:
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            logger.debug(
                f"Cache hit for query (hits: {self._cache_hits}, "
                f"misses: {self._cache_misses})"
            )
            return (self._embedding_cache[cache_key], 0.0)
        self._cache_misses += 1
        (embeddings, cost) = self._generate_embeddings([text])
        embedding = embeddings[0]
        if len(self._embedding_cache) < 1000:
            self._embedding_cache[cache_key] = embedding
            self._save_embedding_cache()
            logger.debug(
                f"Cached new embedding (hits: {self._cache_hits}, "
                f"misses: {self._cache_misses})"
            )
        return (embedding, cost)

    def save_index(self, index_path: Path):
        index_path.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(index_path / "embeddings.npy", self.embeddings)
        if self.bm25 is not None:
            with open(index_path / "bm25_index.pickle", "wb") as f:
                pickle.dump(
                    {"bm25": self.bm25, "tokenized_corpus": self.tokenized_corpus}, f
                )
        with open(index_path / "chunks.pickle", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(index_path / "metadata.json", "w") as f:
            json.dump(self.index_metadata, f, indent=2)
        logger.success(f"Hybrid index saved to {index_path}")

    def load_index(self, index_path: Path) -> bool:
        try:
            embeddings_path = index_path / "embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
            bm25_path = index_path / "bm25_index.pickle"
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    bm25_data = pickle.load(f)
                    self.bm25 = bm25_data["bm25"]
                    self.tokenized_corpus = bm25_data["tokenized_corpus"]
            chunks_path = index_path / "chunks.pickle"
            if chunks_path.exists():
                with open(chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
            metadata_path = index_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.index_metadata = json.load(f)
            if self.chunks:
                logger.success(
                    f"Loaded hybrid index with {len(self.chunks)} chunks "
                    f"from {index_path}"
                )
                return True
            else:
                logger.warning("No chunks found in index")
                return False
        except FileNotFoundError as e:
            logger.error(f"Index file not found: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied loading index: {e}")
            return False
        except (pickle.PickleError, json.JSONDecodeError) as e:
            logger.error(f"Index file corrupted: {e}")
            return False
        except OSError as e:
            logger.error(f"System error loading index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        if not self.chunks:
            return {"status": "empty"}
        stats = {
            "total_chunks": len(self.chunks),
            "has_bm25_index": self.bm25 is not None,
            "has_vector_index": self.embeddings is not None,
            "embedding_model": self.index_metadata.get("embedding_model"),
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "chunk_types": list(
                set((chunk[1].get("chunk_type", "unknown") for chunk in self.chunks))
            ),
            "topic_categories": list(
                set(
                    (chunk[1].get("topic_category", "unknown") for chunk in self.chunks)
                )
            ),
            "embedding_cache": {
                "cached_embeddings": len(self._embedding_cache),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": round(
                    self._cache_hits / max(1, self._cache_hits + self._cache_misses), 3
                ),
            },
        }
        if self.embeddings is not None:
            stats["vector_dimension"] = self.embeddings.shape[1]
            stats["index_size_mb"] = self.embeddings.nbytes / (1024 * 1024)
        return stats

    def _load_embedding_cache(self) -> dict:
        try:
            if self._cache_file.exists():
                with open(self._cache_file, "rb") as f:
                    cache = pickle.load(f)
                logger.debug(f"Loaded {len(cache)} cached embeddings from disk")
                return cache
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
        return {}

    def _save_embedding_cache(self):
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "wb") as f:
                pickle.dump(self._embedding_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def clear_embedding_cache(self):
        self._embedding_cache.clear()
        if self._cache_file.exists():
            self._cache_file.unlink()
        logger.info("Embedding cache cleared")
