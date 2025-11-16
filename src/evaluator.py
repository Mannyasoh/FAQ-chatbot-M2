import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import Any, Dict, List

import openai
from loguru import logger
from openai import OpenAI

from src.config import RAGConfig
from src.costs import estimate_cost_usd

logger.add("logs/evaluator.log", rotation="10 MB", level="INFO", serialize=True)
logger.add("logs/evaluator_error.log", rotation="10 MB", level="ERROR", serialize=True)


class AnswerEvaluator:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from evaluator_prompt.txt file."""
        prompt_file = os.path.join(os.path.dirname(__file__), "evaluator_prompt.txt")
        try:
            with open(prompt_file, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return (
                "You are an expert evaluator. Rate RAG responses 0-10 based on: "
                "RELEVANCE (0-3), ACCURACY (0-4), COMPLETENESS (0-2), CLARITY (0-1). "
                'Return only valid JSON: {"total_score": X, "detailed_scores": '
                '{"relevance": X, "accuracy": X, "completeness": X, "clarity": X}, '
                '"reasoning": "explanation", "strengths": ["list"], '
                '"weaknesses": ["list"], "suggestions": ["list"]}'
            )

    def evaluate_answer(
        self,
        user_question: str,
        system_answer: str,
        chunks_related: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        logger.info(f"Evaluating answer for question: {user_question[:50]}...")
        context_summary = self._summarize_chunks(chunks_related)
        evaluation_prompt = self._create_evaluation_prompt(
            user_question, system_answer, context_summary, chunks_related
        )
        try:
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            evaluation_text = response.choices[0].message.content.strip()
            try:
                evaluation_result = json.loads(evaluation_text)
                if (
                    not isinstance(evaluation_result.get("total_score"), (int, float))
                    or evaluation_result.get("total_score") < 0
                    or evaluation_result.get("total_score") > 10
                ):
                    raise ValueError("Invalid total_score")
                score = evaluation_result["total_score"]
                logger.success(f"Successfully evaluated answer (score: {score}/10)")
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse evaluation JSON, using fallback")
                evaluation_result = self._parse_fallback_evaluation(evaluation_text)
            evaluation_result["metadata"] = {
                "evaluator_model": self.config.generation_model,
                "chunks_count": len(chunks_related),
                "evaluation_cost": estimate_cost_usd(
                    self.config.generation_model,
                    int(len(evaluation_prompt.split()) * 1.3),
                    int(len(evaluation_text.split()) * 1.3),
                ),
            }
            return evaluation_result
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed during evaluation: {e}")
            return {
                "total_score": 0,
                "detailed_scores": {
                    "relevance": 0,
                    "accuracy": 0,
                    "completeness": 0,
                    "clarity": 0,
                },
                "reasoning": "Evaluation failed: Authentication error",
                "strengths": [],
                "weaknesses": ["Technical evaluation error - authentication"],
                "suggestions": ["Check API key configuration"],
                "metadata": {
                    "evaluator_model": self.config.generation_model,
                    "chunks_count": len(chunks_related),
                    "error": "Authentication failed",
                },
            }
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded during evaluation: {e}")
            return {
                "total_score": 0,
                "detailed_scores": {
                    "relevance": 0,
                    "accuracy": 0,
                    "completeness": 0,
                    "clarity": 0,
                },
                "reasoning": "Evaluation failed: Rate limit exceeded",
                "strengths": [],
                "weaknesses": ["Technical evaluation error - rate limit"],
                "suggestions": ["Try evaluation again later"],
                "metadata": {
                    "evaluator_model": self.config.generation_model,
                    "chunks_count": len(chunks_related),
                    "error": "Rate limit exceeded",
                },
            }
        except openai.APIError as e:
            logger.error(f"OpenAI API error during evaluation: {e}")
            return {
                "total_score": 0,
                "detailed_scores": {
                    "relevance": 0,
                    "accuracy": 0,
                    "completeness": 0,
                    "clarity": 0,
                },
                "reasoning": "Evaluation failed: API error",
                "strengths": [],
                "weaknesses": ["Technical evaluation error - API"],
                "suggestions": ["Retry evaluation"],
                "metadata": {
                    "evaluator_model": self.config.generation_model,
                    "chunks_count": len(chunks_related),
                    "error": f"API error: {str(e)}",
                },
            }
        except OSError as e:
            logger.error(f"Network error during evaluation: {e}")
            return {
                "total_score": 0,
                "detailed_scores": {
                    "relevance": 0,
                    "accuracy": 0,
                    "completeness": 0,
                    "clarity": 0,
                },
                "reasoning": "Evaluation failed: Network error",
                "strengths": [],
                "weaknesses": ["Technical evaluation error - network"],
                "suggestions": ["Check internet connection and retry"],
                "metadata": {
                    "evaluator_model": self.config.generation_model,
                    "chunks_count": len(chunks_related),
                    "error": f"Network error: {str(e)}",
                },
            }

    def _summarize_chunks(self, chunks_related: List[Dict[str, Any]]) -> str:
        if not chunks_related:
            return "No relevant chunks retrieved."
        summary_parts = []
        for i, chunk in enumerate(chunks_related):
            chunk_type = chunk.get("chunk_type", "unknown")
            similarity = chunk.get("similarity_score", 0)
            preview = chunk.get("content_preview", "")
            summary_parts.append(
                f"Chunk {i + 1} ({chunk_type}, {similarity:.3f}): {preview}"
            )
        return "\n".join(summary_parts)

    def _create_evaluation_prompt(
        self,
        question: str,
        answer: str,
        context_summary: str,
        chunks_related: List[Dict[str, Any]],
    ) -> str:
        return (
            f"Evaluate this RAG response:\n\n"
            f"QUESTION: {question}\n"
            f"ANSWER: {answer}\n"
            f"CHUNKS ({len(chunks_related)}): {context_summary}\n\n"
            f"Rate 0-10 total:\n"
            f"- RELEVANCE (0-3): Chunks relevant to question?\n"
            f"- ACCURACY (0-4): Answer factually correct from context?\n"
            f"- COMPLETENESS (0-2): Fully addresses question?\n"
            f"- CLARITY (0-1): Clear and well-structured?\n\n"
            f"Return valid JSON with total_score, detailed_scores, "
            f"reasoning, strengths, weaknesses, suggestions."
        )

    def _parse_fallback_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        import re

        score_match = re.search(
            "(?:total.?score|score)[:\\s]*([0-9]+(?:\\.[0-9]+)?)",
            evaluation_text.lower(),
        )
        total_score = min(10.0, float(score_match.group(1)) if score_match else 5.0)
        return {
            "total_score": total_score,
            "detailed_scores": {
                "relevance": round(total_score * 0.3),
                "accuracy": round(total_score * 0.4),
                "completeness": round(total_score * 0.2),
                "clarity": round(total_score * 0.1),
            },
            "reasoning": evaluation_text,
            "strengths": ["Unable to parse"],
            "weaknesses": ["Unable to parse"],
            "suggestions": ["Unable to parse"],
        }

    def evaluate_response_json(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        user_question = response_json.get("user_question", "")
        system_answer = response_json.get("system_answer", "")
        chunks_related = response_json.get("chunks_related", [])
        evaluation = self.evaluate_answer(user_question, system_answer, chunks_related)
        evaluation["original_response"] = {
            "question": user_question,
            "answer": system_answer,
            "chunks_count": len(chunks_related),
        }
        return evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question")
    parser.add_argument("--answer")
    parser.add_argument("--response-file", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    try:
        config = RAGConfig.from_env()
        evaluator = AnswerEvaluator(config)
        if args.response_file:
            with open(args.response_file, "r") as f:
                response_data = json.load(f)
            if isinstance(response_data, list):
                response_json = response_data[0]
            else:
                response_json = response_data
            evaluation = evaluator.evaluate_response_json(response_json)
        elif args.question and args.answer:
            evaluation = evaluator.evaluate_answer(args.question, args.answer, [])
        else:
            print(
                "Error: Either --response-file or both --question and --answer required"
            )
            return
        if args.output:
            with open(args.output, "w") as f:
                json.dump(evaluation, f, indent=2)
            logger.success(f"Evaluation saved to {args.output}")
            print(f"Evaluation saved to {args.output}")
        else:
            print(json.dumps(evaluation, indent=2))
        score = evaluation.get("total_score", 0)
        detailed = evaluation.get("detailed_scores", {})
        rel = detailed.get("relevance", 0)
        acc = detailed.get("accuracy", 0)
        comp = detailed.get("completeness", 0)
        clar = detailed.get("clarity", 0)
        print(f"Score: {score}/10 (R:{rel}/3 A:{acc}/4 C:{comp}/2 Cl:{clar}/1)")
    except (ValueError, FileNotFoundError, PermissionError) as e:
        logger.error(f"Configuration or file error: {e}")
        print(f"Error: {str(e)}")
    except (openai.AuthenticationError, openai.RateLimitError, openai.APIError) as e:
        logger.error(f"OpenAI API error: {e}")
        print(f"OpenAI API Error: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Evaluation process interrupted by user")
        print("Evaluation interrupted")
    except OSError as e:
        logger.error(f"System error: {e}")
        print(f"System Error: {str(e)}")


if __name__ == "__main__":
    main()
