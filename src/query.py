import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import openai
from loguru import logger

from src.config import RAGConfig
from src.generator import AnswerGenerator
from src.hybrid_search import HybridSearchStore
from src.metrics import MetricsCollector, QueryMetrics, SafetyLimits, time_function

logger.add("logs/query.log", rotation="10 MB", level="INFO", serialize=True)
logger.add("logs/query_error.log", rotation="10 MB", level="ERROR", serialize=True)


class FAQChatbot:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.hybrid_store = HybridSearchStore(config)
        self.answer_generator = AnswerGenerator(config)
        self.metrics_collector = MetricsCollector(str(config.metrics_file))
        self.safety_limits = SafetyLimits(
            max_tokens_per_request=config.max_tokens,
            max_chunks_per_query=config.max_chunks_per_query,
            max_cost_per_query=config.max_cost_per_query,
            max_daily_cost=config.max_daily_cost,
        )
        if not self.hybrid_store.load_index(config.index_dir):
            raise RuntimeError(
                "No hybrid index found. Please run build_index.py first."
            )

    @time_function
    def search_chunks(self, query: str):
        return self.hybrid_store.search(
            query, self.config.search_k, search_type=self.config.search_type
        )

    @time_function
    def generate_answer(self, query: str, chunks):
        return self.answer_generator.generate_answer(query, chunks)

    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        if not self.safety_limits.check_daily_cost_limit(self.metrics_collector):
            raise RuntimeError("Daily cost limit exceeded")
        try:
            (search_result, search_latency) = self.search_chunks(question)
            (chunks, search_cost) = search_result
            if not self.safety_limits.check_chunk_limit(len(chunks)):
                chunks = chunks[: self.safety_limits.max_chunks_per_query]
            (gen_result, generation_latency) = self.generate_answer(question, chunks)
            response = gen_result
            total_latency = (time.time() - start_time) * 1000
            generation_cost = response["metadata"].get("generation_cost", 0.0)
            total_cost = search_cost + generation_cost
            if not self.safety_limits.check_cost_limit(total_cost):
                raise RuntimeError(f"Query cost ${total_cost:.6f} exceeds limit")
            metrics = QueryMetrics(
                timestamp=datetime.now().isoformat(),
                query=question,
                chunks_retrieved=len(chunks),
                search_latency_ms=search_latency,
                generation_latency_ms=generation_latency,
                total_latency_ms=total_latency,
                embedding_tokens=int(len(question.split()) * 1.3),
                generation_tokens_in=response["metadata"]["generation_tokens"]["input"],
                generation_tokens_out=response["metadata"]["generation_tokens"][
                    "output"
                ],
                embedding_cost=search_cost,
                generation_cost=generation_cost,
                total_cost=total_cost,
                chunks_used=response["metadata"]["chunks_used"],
                answer_length=len(response["system_answer"]),
            )
            self.metrics_collector.log_query_metrics(metrics)
            if isinstance(response, dict):
                response["metadata"]["search_cost"] = search_cost
                response["metadata"]["total_cost"] = total_cost
                response["metadata"]["search_latency_ms"] = round(search_latency, 2)
                response["metadata"]["generation_latency_ms"] = round(
                    generation_latency, 2
                )
                response["metadata"]["total_latency_ms"] = round(total_latency, 2)
                response["metadata"]["search_strategy"] = self.config.search_type
                response["chunks_related"] = [
                    {
                        "chunk_id": source["chunk_id"],
                        "chunk_type": source["chunk_type"],
                        "similarity_score": source["similarity_score"],
                        "content_preview": source["content_preview"],
                        "exact_matches": source.get("exact_match_phrases", []),
                        "document_section": source.get("document_section", "Unknown"),
                        "line_numbers": source.get("line_numbers", {}),
                    }
                    for source in response.get("sources", [])
                ]
            return response
        except Exception as e:
            error_metrics = QueryMetrics(
                timestamp=datetime.now().isoformat(),
                query=question,
                chunks_retrieved=0,
                search_latency_ms=0,
                generation_latency_ms=0,
                total_latency_ms=(time.time() - start_time) * 1000,
                embedding_tokens=0,
                generation_tokens_in=0,
                generation_tokens_out=0,
                embedding_cost=0.0,
                generation_cost=0.0,
                total_cost=0.0,
                chunks_used=0,
                answer_length=0,
            )
            self.metrics_collector.log_query_metrics(error_metrics)
            return {
                "user_question": question,
                "system_answer": (
                    f"I encountered an error processing your question: {str(e)}"
                ),
                "chunks_related": [],
                "metadata": {"error": str(e), "timestamp": datetime.now().isoformat()},
            }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "hybrid_store": self.hybrid_store.get_stats(),
            "metrics": self.metrics_collector.get_summary_stats(),
            "config": {
                "embedding_model": self.config.embedding_model,
                "generation_model": self.config.generation_model,
                "max_chunks": self.config.max_chunks_per_query,
                "search_k": self.config.search_k,
                "search_type": self.config.search_type,
                "bm25_weight": self.config.bm25_weight,
                "vector_weight": self.config.vector_weight,
            },
        }


def main():
    args = _parse_args()
    try:
        config = RAGConfig.from_env()
        chatbot = _setup_chatbot(config)

        if args.stats:
            _handle_stats(chatbot)
        elif args.interactive:
            _handle_interactive_mode(chatbot)
        else:
            _handle_single_query(chatbot, args)

    except (ValueError, FileNotFoundError, PermissionError) as e:
        _handle_error("Configuration or file error", e, 1)
    except (openai.AuthenticationError, openai.RateLimitError, openai.APIError) as e:
        _handle_error("OpenAI API error", e, 1)
    except KeyboardInterrupt:
        logger.info("Query process interrupted by user")
        print("Query interrupted")
        sys.exit(1)
    except OSError as e:
        _handle_error("System error", e, 1)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", nargs="?", help="Question to ask (required unless using --stats)"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if not args.stats and not args.interactive and (not args.question):
        parser.error(
            "Question argument is required unless using --stats or --interactive"
        )

    return args


def _setup_chatbot(config):
    logger.info("Loading FAQ system...")
    chatbot = FAQChatbot(config)
    logger.success("FAQ system ready!")
    return chatbot


def _handle_stats(chatbot):
    stats = chatbot.get_stats()
    print(json.dumps(stats, indent=2))


def _handle_interactive_mode(chatbot):
    print("\nInteractive mode. Type 'quit' to exit.")
    while True:
        question = input("\nAsk a question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            logger.info("Interactive session ended by user")
            break
        if not question:
            continue

        _process_interactive_query(chatbot, question)


def _process_interactive_query(chatbot, question):
    logger.info(f"Processing query: {question}")
    print("Processing...")
    print(f"\nQ: {question}")
    print("A: ", end="", flush=True)

    def stream_callback(content):
        print(content, end="", flush=True)

    chatbot.answer_generator.set_stream_callback(stream_callback)
    response = chatbot.query(question)
    chatbot.answer_generator.set_stream_callback(None)
    print("\n")

    _display_response_details(response)


def _display_response_details(response):
    sources = response.get("chunks_related", [])
    print(f"\nBased on {len(sources)} relevant sources:")

    for i, source in enumerate(sources[:3], 1):
        _display_source_info(i, source)

    if "validation_notes" in response:
        print(f"Validation: {response['validation_notes']}")

    _display_metadata(response.get("metadata", {}))


def _display_source_info(index, source):
    section = source.get("document_section", "Unknown section")
    chunk_id = source.get("chunk_id", "unknown")
    similarity = source.get("similarity_score", 0)
    line_info = source.get("line_numbers", {})

    print(f"  [{index}] {section} (ID: {chunk_id})")
    print(f"      Relevance: {similarity:.3f}")

    if line_info:
        print(
            f"      Location: Lines {line_info.get('start', '?')}"
            f"-{line_info.get('end', '?')}"
        )

    exact_matches = source.get("exact_matches", [])
    if exact_matches:
        print(f"      Exact matches: {', '.join(exact_matches[:3])}")

    preview = source.get("content_preview", "")
    if preview:
        print(f"      Preview: {preview[:100]}...")
    print()


def _display_metadata(metadata):
    if metadata.get("total_cost"):
        cost = metadata["total_cost"]
        total_latency = metadata.get("total_latency_ms", 0)
        confidence = metadata.get("confidence_score", 0)
        print(
            f"\nCost: ${cost:.6f} | Latency: {total_latency:.1f}ms | "
            f"Confidence: {confidence:.3f}"
        )


def _handle_single_query(chatbot, args):
    logger.info(f"Processing single query: {args.question}")
    response = chatbot.query(args.question)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(response, f, indent=2)
        logger.success(f"Response saved to {args.output}")
        print(f"Response saved to {args.output}")
    else:
        print(json.dumps(response, indent=2))


def _handle_error(error_type, error, exit_code):
    logger.error(f"{error_type}: {error}")
    print(
        f"Error: {str(error)}"
        if error_type.startswith("Configuration")
        else f"{error_type}: {str(error)}"
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
