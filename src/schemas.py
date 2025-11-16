import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402


class SourceCitation(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the source chunk")
    chunk_type: str = Field(
        ..., description="Type of chunk (qa_pair, topic_section, etc.)"
    )
    content_preview: str = Field(..., description="Preview of the source content")
    similarity_score: float = Field(..., description="Relevance score to the query")
    topic_category: Optional[str] = Field(None, description="Topic categorization")
    document_section: Optional[str] = Field(
        None, description="Section of document this came from"
    )
    line_numbers: Optional[Dict[str, int]] = Field(
        None, description="Start and end line numbers in source"
    )
    exact_match_phrases: List[str] = Field(
        default_factory=list, description="Exact phrases from source"
    )


class QueryMetadata(BaseModel):
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_strategy: str = Field(
        ..., description="Search strategy used (hybrid, bm25, vector)"
    )
    generation_tokens: Dict[str, int] = Field(
        ..., description="Token usage for generation"
    )
    search_latency_ms: float = Field(..., description="Search latency in milliseconds")
    generation_latency_ms: float = Field(
        ..., description="Generation latency in milliseconds"
    )
    total_latency_ms: float = Field(
        ..., description="Total query latency in milliseconds"
    )
    search_cost: float = Field(..., description="Cost of embedding search")
    generation_cost: float = Field(..., description="Cost of answer generation")
    total_cost: float = Field(..., description="Total query cost")
    confidence_score: float = Field(..., description="Confidence in the answer (0-1)")
    timestamp: str = Field(..., description="ISO timestamp of query processing")


class RAGResponse(BaseModel):
    user_question: str = Field(..., description="The original user question")
    system_answer: str = Field(..., description="Generated answer from the system")
    sources: List[SourceCitation] = Field(
        ..., description="Source citations for answer validation"
    )
    metadata: QueryMetadata = Field(..., description="Processing metadata")
    answer_confidence: float = Field(
        ..., description="Overall confidence in the answer (0-1)"
    )
    validation_notes: Optional[str] = Field(
        None, description="Notes about answer validation"
    )


class EvaluationScores(BaseModel):
    relevance: int = Field(
        ..., ge=0, le=3, description="Relevance of sources to question (0-3)"
    )
    accuracy: int = Field(
        ..., ge=0, le=4, description="Factual accuracy based on sources (0-4)"
    )
    completeness: int = Field(
        ..., ge=0, le=2, description="Completeness of answer (0-2)"
    )
    clarity: int = Field(..., ge=0, le=1, description="Clarity and structure (0-1)")


class EvaluationResponse(BaseModel):
    total_score: float = Field(
        ..., ge=0, le=10, description="Total evaluation score (0-10)"
    )
    detailed_scores: EvaluationScores = Field(
        ..., description="Breakdown of evaluation criteria"
    )
    reasoning: str = Field(..., description="Detailed reasoning for the evaluation")
    strengths: List[str] = Field(
        ..., description="Identified strengths in the response"
    )
    weaknesses: List[str] = Field(
        ..., description="Identified weaknesses in the response"
    )
    suggestions: List[str] = Field(..., description="Suggestions for improvement")
    citation_analysis: Dict[str, Any] = Field(
        default_factory=dict, description="Analysis of source citations"
    )
    hallucination_risk: str = Field(
        ..., description="Assessment of hallucination risk (low/medium/high)"
    )


class SystemStats(BaseModel):
    index_stats: Dict[str, Any] = Field(..., description="Search index statistics")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    configuration: Dict[str, Any] = Field(..., description="System configuration")
    usage_stats: Dict[str, Any] = Field(..., description="Usage statistics")


class MetadataFilter(BaseModel):
    chunk_type: Optional[str] = Field(None, description="Filter by chunk type")
    topic_category: Optional[str] = Field(None, description="Filter by topic category")
    document_section: Optional[str] = Field(
        None, description="Filter by document section"
    )
    min_similarity: Optional[float] = Field(
        None, description="Minimum similarity threshold"
    )
    has_exact_match: Optional[bool] = Field(
        None, description="Filter for exact phrase matches"
    )


class APIErrorResponse(BaseModel):
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = Field(
        None, description="Request identifier for tracking"
    )
