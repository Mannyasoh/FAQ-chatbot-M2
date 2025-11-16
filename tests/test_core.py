import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.chunking import TextChunker
from src.config import RAGConfig
from src.costs import estimate_cost_usd, estimate_embedding_cost
from src.metrics import MetricsCollector, SafetyLimits


class TestConfig:
    def test_config_from_env(self):
        """Test configuration loading from environment variables."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "CHUNK_SIZE": "300",
                "SEARCH_K": "3",
                "GENERATION_MODEL": "gpt-4o",
            },
        ):
            config = RAGConfig.from_env()
            assert config.openai_api_key == "test-key"
            assert config.chunk_size == 300
            assert config.search_k == 3
            assert config.generation_model == "gpt-4o"

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = RAGConfig(openai_api_key="test-key")
        assert config.validate_config() is True

    def test_config_validation_missing_api_key(self):
        """Test validation failure with missing API key."""
        config = RAGConfig(openai_api_key="")
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            config.validate_config()

    def test_config_validation_invalid_chunk_size(self):
        """Test validation failure with invalid chunk size."""
        config = RAGConfig(openai_api_key="test-key", chunk_size=10, min_chunk_size=50)
        with pytest.raises(ValueError, match="chunk_size must be at least"):
            config.validate_config()

    def test_hybrid_search_config(self):
        """Test hybrid search configuration parameters."""
        config = RAGConfig.from_env()
        assert hasattr(config, "bm25_weight")
        assert hasattr(config, "vector_weight")
        assert hasattr(config, "search_type")
        assert config.bm25_weight + config.vector_weight == 1.0


class TestTextChunker:
    def test_chunker_initialization(self):
        """Test chunker initialization with parameters."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 50
        assert chunker.min_chunk_size == 50

    def test_chunk_simple_text(self):
        """Test chunking of simple text."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a simple test document. " * 10  # ~320 chars
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, tuple) and len(chunk) == 2 for chunk in chunks)
        assert all(isinstance(chunk[0], str) for chunk in chunks)
        assert all(isinstance(chunk[1], dict) for chunk in chunks)

    def test_qa_structure_recognition(self):
        """Test Q&A structure recognition in chunking."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        qa_text = """Q: What is the vacation policy?
A: Employees get 15 days of vacation annually.

Q: How do I submit a timesheet?
A: Use the online portal to submit timesheets."""
        chunks = chunker.chunk_text(qa_text)

        # Should have chunks with Q&A content
        assert len(chunks) >= 1

        # Check that Q&A content is preserved
        has_qa_content = any("Q:" in chunk[0] and "A:" in chunk[0] for chunk in chunks)
        assert has_qa_content, "Chunks should contain Q&A content"

    def test_minimum_chunk_length(self):
        """Test that chunks meet minimum length requirements."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=30)
        text = "Short text."
        chunks = chunker.chunk_text(text)

        # Very short text might not produce any chunks if below min_chunk_size
        for chunk_text, _ in chunks:
            assert len(chunk_text) >= chunker.min_chunk_size


class TestCosts:
    def test_estimate_cost_usd_basic(self):
        """Test basic cost calculation."""
        cost = estimate_cost_usd("gpt-4o-mini", 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)
        assert cost < 1.0  # Should be reasonable for small token counts

    def test_estimate_embedding_cost(self):
        """Test embedding cost calculation."""
        cost = estimate_embedding_cost("text-embedding-3-small", 1000)
        assert cost > 0
        assert isinstance(cost, float)
        assert cost < 0.1  # Embedding costs should be very low

    def test_cost_validation_negative_tokens(self):
        """Test cost calculation rejects negative token counts."""
        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            estimate_cost_usd("gpt-4o-mini", -100, 500)

    def test_cost_unknown_model_fallback(self):
        """Test cost calculation with unknown model uses fallback."""
        cost = estimate_cost_usd("unknown-model", 1000, 500)
        assert cost > 0  # Should use fallback pricing

    def test_cost_with_cached_tokens(self):
        """Test cost calculation with cached input tokens."""
        cost_no_cache = estimate_cost_usd("gpt-4o-mini", 1000, 500, 0)
        cost_with_cache = estimate_cost_usd("gpt-4o-mini", 1000, 500, 200)
        assert cost_with_cache < cost_no_cache  # Cache should reduce cost


class TestMetricsAndSafety:
    def test_safety_limits_initialization(self):
        """Test safety limits initialization."""
        limits = SafetyLimits(
            max_tokens_per_request=1000,
            max_chunks_per_query=5,
            max_cost_per_query=2.0,
            max_daily_cost=100.0,
        )

        assert limits.max_tokens_per_request == 1000
        assert limits.max_chunks_per_query == 5
        assert limits.max_cost_per_query == 2.0
        assert limits.max_daily_cost == 100.0

    def test_safety_limits_checks(self):
        """Test safety limit validation methods."""
        limits = SafetyLimits(max_tokens_per_request=1000, max_chunks_per_query=5)

        assert limits.check_token_limit(500) is True
        assert limits.check_token_limit(1500) is False
        assert limits.check_chunk_limit(3) is True
        assert limits.check_chunk_limit(10) is False

    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization and CSV creation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            metrics_file = f.name

        try:
            collector = MetricsCollector(metrics_file)

            # Test that CSV file is created with headers
            assert Path(metrics_file).exists()

            # Test summary with empty file
            stats = collector.get_summary_stats()
            assert isinstance(stats, dict)
            assert stats.get("total_queries", 0) == 0

        finally:
            Path(metrics_file).unlink(missing_ok=True)

    def test_metrics_collector_empty_stats(self):
        """Test metrics collector returns empty stats for new file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            metrics_file = f.name

        try:
            collector = MetricsCollector(metrics_file)
            stats = collector.get_summary_stats()

            # Empty file should return empty dict or zero values
            if stats:
                assert stats.get("total_queries", 0) == 0
                assert stats.get("total_cost_usd", 0) == 0

        finally:
            Path(metrics_file).unlink(missing_ok=True)


class TestHybridSearchMocked:
    """Test hybrid search functionality with mocked OpenAI calls."""

    @patch("openai.OpenAI")
    def test_hybrid_search_initialization(self, mock_openai):
        """Test hybrid search store initialization."""
        from src.hybrid_search import HybridSearchStore

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        config = RAGConfig(openai_api_key="test-key")
        store = HybridSearchStore(config)

        assert store.config == config
        assert store.chunks == []
        assert store.embeddings is None
        assert store.bm25 is None

    def test_hybrid_search_add_chunks_no_api(self):
        """Test adding chunks to hybrid search store without API calls."""
        from src.hybrid_search import HybridSearchStore

        config = RAGConfig(openai_api_key="test-key")
        store = HybridSearchStore(config)

        # Test BM25 components without requiring embeddings
        chunks = [
            (
                "Test chunk 1 about vacation policy",
                {"chunk_id": "1", "chunk_type": "qa_pair"},
            ),
            (
                "Test chunk 2 about payroll",
                {"chunk_id": "2", "chunk_type": "topic_section"},
            ),
        ]

        # Set chunks directly and test BM25 functionality
        store.chunks = chunks
        texts = [chunk[0] for chunk in chunks]
        store.tokenized_corpus = [store._tokenize_text(text) for text in texts]

        # Test that BM25 can be created
        from rank_bm25 import BM25Okapi

        store.bm25 = BM25Okapi(store.tokenized_corpus)

        assert len(store.chunks) == 2
        assert store.bm25 is not None
        assert len(store.tokenized_corpus) == 2

        # Test BM25 scoring
        query = "vacation policy"
        query_tokens = store._tokenize_text(query)
        scores = store.bm25.get_scores(query_tokens)
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)

    def test_bm25_tokenization(self):
        """Test BM25 tokenization functionality."""
        from src.hybrid_search import HybridSearchStore

        config = RAGConfig(openai_api_key="test-key")
        store = HybridSearchStore(config)

        text = "This is a test document with punctuation!"
        tokens = store._tokenize_text(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        assert "punctuation" in tokens  # Should extract words without punctuation

    def test_metadata_enhancement(self):
        """Test metadata enhancement for search optimization."""
        from src.hybrid_search import HybridSearchStore

        config = RAGConfig(openai_api_key="test-key")
        store = HybridSearchStore(config)

        chunks = [
            ("Q: What is the vacation policy? A: 15 days annually.", {"chunk_id": "1"}),
            ("Payroll processing occurs bi-weekly.", {"chunk_id": "2"}),
        ]

        store.chunks = chunks
        store._enhance_metadata()

        # Check enhanced metadata
        for chunk_text, metadata in store.chunks:
            assert "text_length" in metadata
            assert "word_count" in metadata
            assert "has_question" in metadata
            assert "topic_category" in metadata


class TestIntegration:
    """Simple integration tests using mocks."""

    def test_sample_outputs_json_structure(self):
        """Test that sample outputs have the required JSON structure."""
        import json

        sample_path = Path("outputs/sample_queries.json")
        if not sample_path.exists():
            pytest.skip("Sample outputs file not found")

        with open(sample_path, "r") as f:
            samples = json.load(f)

        assert isinstance(samples, list)
        assert len(samples) >= 3

        required_keys = {"user_question", "system_answer", "chunks_related"}

        for sample in samples:
            assert isinstance(sample, dict)
            assert all(key in sample for key in required_keys)

            # Test chunks_related structure
            chunks_related = sample["chunks_related"]
            assert isinstance(chunks_related, list)

            for chunk in chunks_related:
                assert isinstance(chunk, dict)
                chunk_required = {"chunk_id", "similarity_score", "content_preview"}
                assert all(key in chunk for key in chunk_required)

    def test_faq_document_structure(self):
        """Test FAQ document meets basic requirements."""
        doc_path = Path("data/faq_document.txt")
        if not doc_path.exists():
            pytest.skip("FAQ document not found")

        with open(doc_path, "r") as f:
            content = f.read()

        # Basic requirements
        assert len(content) > 1000, "Document should be at least 1000 characters"

        # Should have Q&A structure
        assert "Q:" in content, "Document should contain questions"
        assert "A:" in content, "Document should contain answers"

        # Count Q&A pairs
        qa_pairs = content.count("Q:")
        assert qa_pairs >= 10, f"Should have at least 10 Q&A pairs, found {qa_pairs}"

    def test_project_structure_completeness(self):
        """Test that all required files exist."""
        required_files = [
            "src/__init__.py",
            "src/build_index.py",
            "src/query.py",
            "src/chunking.py",
            "src/hybrid_search.py",
            "src/generator.py",
            "src/config.py",
            "src/costs.py",
            "src/metrics.py",
            "data/faq_document.txt",
            "outputs/sample_queries.json",
            "requirements.txt",
            ".env.example",
            "README.md",
        ]

        for file_path in required_files:
            assert Path(file_path).exists(), f"Required file missing: {file_path}"

    def test_end_to_end_pipeline_no_api(self):
        """Test end-to-end pipeline without external calls."""
        from src.chunking import TextChunker
        from src.hybrid_search import HybridSearchStore

        # Test document processing pipeline
        sample_doc = """Q: What is the vacation policy?
A: Employees receive 15 days of vacation annually.

Q: How do I request time off?
A: Submit a request through HR portal.

BENEFITS AND COMPENSATION

Health insurance coverage begins on the first day."""

        # Test chunking
        chunker = TextChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk_text(sample_doc)

        assert len(chunks) >= 1

        # Test that chunks contain meaningful content
        total_content = " ".join(chunk[0] for chunk in chunks)
        assert "vacation policy" in total_content
        assert "request time off" in total_content

        # Test BM25 functionality without requiring embeddings
        config = RAGConfig(openai_api_key="test-key")
        store = HybridSearchStore(config)

        # Set up BM25 without embeddings
        store.chunks = chunks
        texts = [chunk[0] for chunk in chunks]
        store.tokenized_corpus = [store._tokenize_text(text) for text in texts]

        from rank_bm25 import BM25Okapi

        store.bm25 = BM25Okapi(store.tokenized_corpus)

        assert store.bm25 is not None
        assert len(store.tokenized_corpus) > 0


# Utility function for tests
def create_test_config(**kwargs):
    """Create a test configuration with sensible defaults."""
    defaults = {
        "openai_api_key": "test-key",
        "chunk_size": 300,
        "search_k": 3,
        "max_tokens": 1000,
        "bm25_weight": 0.4,
        "vector_weight": 0.6,
    }
    defaults.update(kwargs)
    return RAGConfig(**defaults)


# Test the utility function
def test_create_test_config():
    """Test the test configuration utility."""
    config = create_test_config(chunk_size=500)
    assert config.chunk_size == 500
    assert config.openai_api_key == "test-key"
    assert config.bm25_weight == 0.4
