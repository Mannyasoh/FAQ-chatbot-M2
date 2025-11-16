import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import openai
from loguru import logger

from src.chunking import TextChunker
from src.config import RAGConfig
from src.costs import estimate_embedding_cost
from src.hybrid_search import HybridSearchStore
from src.metrics import SafetyLimits

logger.add("logs/build_index.log", rotation="10 MB", level="INFO", serialize=True)
logger.add("logs/build_error.log", rotation="10 MB", level="ERROR", serialize=True)


def load_document(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.info(
                f"Successfully loaded document: {file_path} ({len(content)} characters)"
            )
            return content
    except FileNotFoundError as e:
        logger.error(f"Document not found: {file_path}")
        raise FileNotFoundError(f"Document not found: {file_path}") from e
    except PermissionError as e:
        logger.error(f"Permission denied reading document: {file_path}")
        raise PermissionError(f"Permission denied reading document: {file_path}") from e
    except UnicodeDecodeError as e:
        logger.error(f"Invalid encoding in document: {file_path}")
        raise UnicodeDecodeError(
            e.encoding,
            e.object,
            e.start,
            e.end,
            f"Invalid encoding in document: {file_path}",
        ) from e
    except OSError as e:
        logger.error(f"OS error reading document: {file_path} - {e}")
        raise OSError(f"OS error reading document: {file_path}") from e


def build_index(
    config: RAGConfig, document_path: Path, force_rebuild: bool = False
) -> dict:
    try:
        logger.info(
            "Starting index build process",
            extra={
                "document_path": str(document_path),
                "force_rebuild": force_rebuild,
                "embedding_model": config.embedding_model,
            },
        )
        config.validate_config()
        logger.debug("Configuration validated successfully")
        chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=50,
        )
        hybrid_store = HybridSearchStore(config)
        safety_limits = SafetyLimits(
            max_chunks_per_query=config.max_chunks_per_query,
            max_cost_per_query=config.max_cost_per_query,
            max_daily_cost=config.max_daily_cost,
        )
        if not force_rebuild and hybrid_store.load_index(config.index_dir):
            logger.info("Existing index found, skipping rebuild")
            return hybrid_store.get_stats()
        logger.info(f"Loading document: {document_path}")
        document_text = load_document(document_path)
        if len(document_text.strip()) < 1000:
            raise ValueError("Document must be at least 1000 characters long")
        logger.info("Starting document chunking")
        document_name = document_path.name
        chunks = chunker.chunk_text(document_text, document_name)
        if len(chunks) < 20:
            logger.warning(f"Only {len(chunks)} chunks generated, adjusting strategy")
            chunker.chunk_size = 300
            chunker.chunk_overlap = 50
            chunks = chunker.chunk_text(document_text)
        logger.info(f"Generated {len(chunks)} chunks")
        total_tokens = sum((chunk[1].get("token_count", 0) for chunk in chunks))
        estimated_cost = estimate_embedding_cost(config.embedding_model, total_tokens)
        if not safety_limits.check_cost_limit(estimated_cost):
            raise ValueError(
                f"Estimated cost ${estimated_cost:.6f} exceeds limit "
                f"${safety_limits.max_cost_per_query}"
            )
        if not safety_limits.check_chunk_limit(len(chunks)):
            logger.warning(
                f"{len(chunks)} chunks exceeds recommended limit of "
                f"{safety_limits.max_chunks_per_query}"
            )
        logger.info(f"Estimated embedding cost: ${estimated_cost:.6f}")
        total_cost = hybrid_store.add_chunks(chunks)
        hybrid_store.save_index(config.index_dir)
        stats = hybrid_store.get_stats()
        logger.success(
            "Index build completed successfully",
            extra={
                "total_chunks": stats["total_chunks"],
                "total_cost": total_cost,
                "index_size_mb": stats.get("index_size_mb", 0),
            },
        )
        return stats
    except ValueError as e:
        logger.error(f"Configuration or validation error: {e}")
        raise
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication failed: {e}")
        raise openai.AuthenticationError("Invalid OpenAI API key") from e
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded: {e}")
        raise openai.RateLimitError(
            "OpenAI rate limit exceeded, please try again later"
        ) from e
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise openai.APIError(f"OpenAI API error: {e}") from e
    except OSError as e:
        logger.error(f"File system error: {e}")
        raise OSError(f"File system error: {e}") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=Path, default=Path("data/faq_document.txt"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        os.makedirs("logs", exist_ok=True)
        config = RAGConfig.from_env()
        if not args.document.exists():
            logger.error(f"Document not found: {args.document}")
            print(f"Error: Document not found at {args.document}")
            sys.exit(1)
        stats = build_index(config, args.document, args.force)
        print(f"Index built successfully! ({stats['total_chunks']} chunks)")
    except (ValueError, FileNotFoundError, PermissionError) as e:
        logger.error(f"Build failed: {e}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    except (openai.AuthenticationError, openai.RateLimitError, openai.APIError) as e:
        logger.error(f"OpenAI API error: {e}")
        print(f"OpenAI API Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Build process interrupted by user")
        print("Build interrupted")
        sys.exit(1)
    except OSError as e:
        logger.error(f"System error: {e}")
        print(f"System Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
