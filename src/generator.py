import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Any, Dict, List, Tuple

import openai
import tiktoken
from loguru import logger
from openai import OpenAI

from src.config import RAGConfig
from src.costs import estimate_cost_usd
from src.schemas import QueryMetadata, RAGResponse, SourceCitation

logger.add("logs/generator.log", rotation="10 MB", level="INFO", serialize=True)
logger.add("logs/generator_error.log", rotation="10 MB", level="ERROR", serialize=True)


class AnswerGenerator:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.tokenizer = tiktoken.encoding_for_model(config.generation_model)
        self._stream_callback = None
        self._stream_callback = None

    def set_stream_callback(self, callback):
        self._stream_callback = callback

    def generate_answer(
        self, query: str, chunks: List[Tuple[str, Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        if not chunks:
            return self._create_empty_response(query)

        prompt, prompt_tokens = self._prepare_prompt(query, chunks)
        try:
            return self._generate_response(query, chunks, prompt, prompt_tokens)
        except (
            openai.AuthenticationError,
            openai.RateLimitError,
            openai.APIError,
            OSError,
        ) as e:
            return self._handle_api_error(query, e)

    def _create_empty_response(self, query: str) -> Dict[str, Any]:
        logger.warning(f"No chunks found for query: {query}")
        return {
            "user_question": query,
            "system_answer": (
                "I couldn't find any relevant information to "
                "answer your question. Please try rephrasing or "
                "asking about a different topic."
            ),
            "chunks_related": [],
            "metadata": {
                "chunks_used": 0,
                "generation_tokens": {"input": 0, "output": 0},
                "generation_cost": 0.0,
                "confidence": 0.0,
            },
        }

    def _prepare_prompt(self, query: str, chunks: List) -> Tuple[str, int]:
        context = self._build_context(chunks)
        prompt = self._create_prompt(query, context)
        prompt_tokens = len(self.tokenizer.encode(prompt))

        if prompt_tokens > self.config.max_tokens - 500:
            logger.warning(
                f"Prompt too long ({prompt_tokens} tokens), truncating context"
            )
            context = self._truncate_context(context, self.config.max_tokens - 1000)
            prompt = self._create_prompt(query, context)
            prompt_tokens = len(self.tokenizer.encode(prompt))

        return prompt, prompt_tokens

    def _generate_response(
        self, query: str, chunks: List, prompt: str, prompt_tokens: int
    ) -> Dict[str, Any]:
        answer, completion_tokens = self._call_openai_stream(prompt)
        generation_cost = estimate_cost_usd(
            self.config.generation_model, prompt_tokens, completion_tokens
        )

        logger.info(
            f"Generated answer for query: {query[:50]}... "
            f"(tokens: {prompt_tokens}→{completion_tokens}, "
            f"cost: ${generation_cost:.6f})"
        )

        sources = self._build_sources(query, chunks)
        confidence = self._calculate_confidence(chunks)
        validation_notes = self._generate_validation_notes(sources, answer)

        query_metadata = QueryMetadata(
            chunks_used=len(chunks),
            search_strategy="hybrid",
            generation_tokens={"input": prompt_tokens, "output": completion_tokens},
            search_latency_ms=0,
            generation_latency_ms=0,
            total_latency_ms=0,
            search_cost=0.0,
            generation_cost=generation_cost,
            total_cost=generation_cost,
            confidence_score=confidence,
            timestamp=datetime.now().isoformat(),
        )

        return RAGResponse(
            user_question=query,
            system_answer=answer,
            sources=sources,
            metadata=query_metadata,
            answer_confidence=confidence,
            validation_notes=validation_notes,
        ).dict()

    def _call_openai_stream(self, prompt: str) -> Tuple[str, int]:
        stream = self.client.chat.completions.create(
            model=self.config.generation_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful HR assistant. Provide "
                        "accurate, helpful answers based only on the "
                        "provided context. If you cannot answer based "
                        "on the context, say so clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=500,
            stream=True,
        )

        answer_parts = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                answer_parts.append(content)
                if (
                    hasattr(self, "_stream_callback")
                    and self._stream_callback is not None
                ):
                    self._stream_callback(content)

        answer = "".join(answer_parts).strip()
        completion_tokens = len(self.tokenizer.encode(answer))
        return answer, completion_tokens

    def _build_sources(self, query: str, chunks: List) -> List[SourceCitation]:
        sources = []
        for chunk_text, metadata, similarity in chunks:
            exact_matches = self._find_exact_matches(query, chunk_text)
            document_section = self._determine_document_section(metadata)

            preview = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text

            source = SourceCitation(
                chunk_id=metadata.get("chunk_id", "unknown"),
                chunk_type=metadata.get("chunk_type", "unknown"),
                content_preview=preview,
                similarity_score=round(similarity, 4),
                topic_category=metadata.get("topic_category", "general"),
                document_section=document_section,
                line_numbers=metadata.get("line_numbers"),
                exact_match_phrases=exact_matches,
            )
            sources.append(source)
        return sources

    def _handle_api_error(self, query: str, error: Exception) -> Dict[str, Any]:
        base_response = {
            "user_question": query,
            "chunks_related": [],
            "metadata": {
                "chunks_used": 0,
                "generation_tokens": {"input": 0, "output": 0},
                "generation_cost": 0.0,
                "confidence": 0.0,
            },
        }

        if isinstance(error, openai.AuthenticationError):
            logger.error(f"OpenAI authentication failed: {error}")
            base_response.update(
                {
                    "system_answer": (
                        "I encountered an authentication error. "
                        "Please check the API key configuration."
                    ),
                    "metadata": {
                        **base_response["metadata"],  # type: ignore[dict-item]
                        "error": "Authentication failed",
                    },
                }
            )
        elif isinstance(error, openai.RateLimitError):
            logger.error(f"OpenAI rate limit exceeded: {error}")
            base_response.update(
                {
                    "system_answer": (
                        "I'm currently experiencing high demand. "
                        "Please try again in a moment."
                    ),
                    "metadata": {
                        **base_response["metadata"],  # type: ignore[dict-item]
                        "error": "Rate limit exceeded",
                    },
                }
            )
        elif isinstance(error, openai.APIError):
            logger.error(f"OpenAI API error: {error}")
            base_response.update(
                {
                    "system_answer": (
                        "I encountered a technical error while "
                        "processing your request. Please try again."
                    ),
                    "metadata": {
                        **base_response["metadata"],  # type: ignore[dict-item]
                        "error": f"API error: {str(error)}",
                    },
                }
            )
        elif isinstance(error, OSError):
            logger.error(f"Network or system error during generation: {error}")
            base_response.update(
                {
                    "system_answer": (
                        "I encountered a connection error. Please "
                        "check your internet connection and try again."
                    ),
                    "metadata": {
                        **base_response["metadata"],  # type: ignore[dict-item]
                        "error": f"Network error: {str(error)}",
                    },
                }
            )

        return base_response

    def _build_context(self, chunks: List[Tuple[str, Dict[str, Any], float]]) -> str:
        context_parts = []
        for i, (chunk_text, metadata, similarity) in enumerate(chunks):
            chunk_type = metadata.get("chunk_type", "unknown")
            if chunk_type == "qa_pair":
                context_parts.append(f"[Context {i + 1}]\n{chunk_text}")
            else:
                topic = metadata.get("topic", "General Information")
                context_parts.append(f"[Context {i + 1} - {topic}]\n{chunk_text}")
        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        prompt = (
            f"Based on the following HR documentation context, please answer "
            f"the user's question. \nBe specific, accurate, and helpful. If "
            f"the information is not in the context, say so clearly.\n\n"
            f"CONTEXT:\n{context}\n\nUSER QUESTION: {query}\n\n"
            f"Please provide a clear, helpful answer based on the context above. "
            f"Include specific details like numbers, dates, or procedures when "
            f"available."
        )
        return prompt

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(context)
        if len(tokens) <= max_tokens:
            return context
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def _calculate_confidence(
        self, chunks: List[Tuple[str, Dict[str, Any], float]]
    ) -> float:
        if not chunks:
            return 0.0
        similarities = [similarity for (_, _, similarity) in chunks]
        avg_similarity = sum(similarities) / len(similarities)
        confidence = avg_similarity * min(1.0, len(chunks) / 3.0)
        return round(confidence, 4)

    def _find_exact_matches(self, query: str, chunk_text: str) -> List[str]:
        import re

        exact_matches = []
        query_lower = query.lower()
        chunk_lower = chunk_text.lower()
        query_words = re.findall("\\b\\w{3,}\\b", query_lower)
        for word in query_words:
            if word in chunk_lower:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(chunk_text)
                exact_matches.extend(matches)
        query_phrases = re.findall("\\b\\w+\\s+\\w+(?:\\s+\\w+)*\\b", query)
        for phrase in query_phrases:
            if len(phrase.split()) >= 2:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                if pattern.search(chunk_text):
                    exact_matches.append(phrase)
        return list(set(exact_matches))

    def _determine_document_section(self, metadata: Dict[str, Any]) -> str:
        chunk_type = metadata.get("chunk_type", "unknown")
        topic_category = metadata.get("topic_category", "general")
        if chunk_type == "qa_pair":
            question = metadata.get("question", "")
            if "vacation" in question.lower() or "time off" in question.lower():
                return "Time Off and Leave Policies"
            elif "health" in question.lower() or "insurance" in question.lower():
                return "Benefits and Insurance"
            elif "payroll" in question.lower() or "pay" in question.lower():
                return "Payroll and Compensation"
            elif "performance" in question.lower() or "review" in question.lower():
                return "Performance Management"
            elif "onboard" in question.lower() or "new employee" in question.lower():
                return "Employee Onboarding"
            else:
                return "General HR Policies"
        if topic_category == "time_off":
            return "Time Off and Leave Policies"
        elif topic_category == "benefits":
            return "Benefits and Insurance"
        elif topic_category == "payroll":
            return "Payroll and Compensation"
        elif topic_category == "onboarding":
            return "Employee Onboarding"
        else:
            return "General HR Information"

    def _generate_validation_notes(
        self, sources: List[SourceCitation], answer: str
    ) -> str:
        notes = []
        if not sources:
            notes.append(
                "WARNING: No source citations found - potential hallucination risk"
            )
        elif len(sources) == 1:
            notes.append("Single source used - consider cross-referencing")
        else:
            notes.append(f"Answer based on {len(sources)} sources from document")
        total_exact_matches = sum(
            (len(source.exact_match_phrases) for source in sources)
        )
        if total_exact_matches == 0:
            notes.append("No exact phrase matches found - verify answer accuracy")
        else:
            notes.append(
                f"Contains {total_exact_matches} exact phrase matches from sources"
            )
        source_types = set((source.chunk_type for source in sources))
        if len(source_types) > 1:
            notes.append(
                f"Sources from multiple chunk types: {', '.join(source_types)}"
            )
        return "; ".join(notes)
