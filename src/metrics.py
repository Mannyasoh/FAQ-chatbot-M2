import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class QueryMetrics:
    timestamp: str
    query: str
    chunks_retrieved: int
    search_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    embedding_tokens: int
    generation_tokens_in: int
    generation_tokens_out: int
    embedding_cost: float
    generation_cost: float
    total_cost: float
    chunks_used: int
    answer_length: int


class MetricsCollector:
    def __init__(self, metrics_file: str = "metrics/rag_metrics.csv"):
        self.metrics_file = metrics_file
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        self._init_csv_file()

    def _init_csv_file(self):
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "query",
                        "chunks_retrieved",
                        "search_latency_ms",
                        "generation_latency_ms",
                        "total_latency_ms",
                        "embedding_tokens",
                        "generation_tokens_in",
                        "generation_tokens_out",
                        "embedding_cost",
                        "generation_cost",
                        "total_cost",
                        "chunks_used",
                        "answer_length",
                    ]
                )

    def log_query_metrics(self, metrics: QueryMetrics):
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    metrics.timestamp,
                    metrics.query,
                    metrics.chunks_retrieved,
                    metrics.search_latency_ms,
                    metrics.generation_latency_ms,
                    metrics.total_latency_ms,
                    metrics.embedding_tokens,
                    metrics.generation_tokens_in,
                    metrics.generation_tokens_out,
                    metrics.embedding_cost,
                    metrics.generation_cost,
                    metrics.total_cost,
                    metrics.chunks_used,
                    metrics.answer_length,
                ]
            )

    def get_summary_stats(self) -> Dict[str, Any]:
        if not os.path.exists(self.metrics_file):
            return {}
        with open(self.metrics_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return {}
        latencies = [
            float(row["total_latency_ms"]) for row in rows if row["total_latency_ms"]
        ]
        costs = [float(row["total_cost"]) for row in rows if row["total_cost"]]
        return {
            "total_queries": len(rows),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)]
            if latencies
            else 0,
            "total_cost_usd": sum(costs),
            "avg_cost_per_query": sum(costs) / len(costs) if costs else 0,
        }


class SafetyLimits:
    def __init__(
        self,
        max_tokens_per_request: int = 8192,
        max_chunks_per_query: int = 10,
        max_cost_per_query: float = 1.0,
        max_daily_cost: float = 50.0,
    ):
        self.max_tokens_per_request = max_tokens_per_request
        self.max_chunks_per_query = max_chunks_per_query
        self.max_cost_per_query = max_cost_per_query
        self.max_daily_cost = max_daily_cost

    def check_token_limit(self, tokens: int) -> bool:
        return tokens <= self.max_tokens_per_request

    def check_chunk_limit(self, chunks: int) -> bool:
        return chunks <= self.max_chunks_per_query

    def check_cost_limit(self, estimated_cost: float) -> bool:
        return estimated_cost <= self.max_cost_per_query

    def check_daily_cost_limit(self, metrics_collector: MetricsCollector) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(metrics_collector.metrics_file):
            return True
        with open(metrics_collector.metrics_file, "r") as f:
            reader = csv.DictReader(f)
            daily_cost = sum(
                (
                    float(row["total_cost"])
                    for row in reader
                    if row["timestamp"].startswith(today) and row["total_cost"]
                )
            )
        return daily_cost < self.max_daily_cost


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return (result, (end_time - start_time) * 1000)

    return wrapper
