# HR FAQ Support Chatbot - RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for answering HR-related questions using vector search and large language models. This system intelligently chunks HR documentation, generates embeddings, and provides accurate answers with full transparency and cost tracking.

## Features

- **Intelligent Document Processing**: Multi-strategy chunking including Q&A structure recognition and semantic segmentation
- **Hybrid Search**: Advanced retrieval combining BM25 keyword search and dense vector similarity using Reciprocal Rank Fusion
- **LLM Answer Generation**: Context-aware answer generation with relevance scoring
- **Cost Monitoring**: Comprehensive tracking of embedding and generation costs
- **Safety Limits**: Built-in guardrails for token limits, chunk limits, and daily cost limits
- **Quality Evaluation**: Optional evaluator agent that scores answer quality (0-10) with detailed reasoning
- **Metrics & Analytics**: Detailed performance metrics including latency, costs, and quality scores

## Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to the project
cd assignment-m2

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 2. Configure API Key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Build the Knowledge Base

```bash
# Process the FAQ document and build vector index
python -m src.build_index --document data/faq_document.txt

# Incase the index exists and you want to rebuild the index forcefully
python -m src.build_index --document data/faq_document.txt --force
```

### 4. Start Asking Questions

```bash
# Interactive mode
python -m src.query --interactive

# Single query
python -m src.query "How do I set up direct deposit?"

# Save response to file
python -m src.query "What happens if I work overtime?" --output response.json
```

## Project Structure

```
assignment-m2/
├── src/                          # Core source code
│   ├── build_index.py           # Data pipeline for building hybrid search index
│   ├── query.py                 # Query pipeline for answering questions
│   ├── chunking.py              # Text chunking strategies
│   ├── hybrid_search.py         # Hybrid search combining BM25 + vector search
│   ├── generator.py             # LLM answer generation
│   ├── evaluator.py             # Optional quality evaluation agent
│   ├── config.py                # Configuration management
│   ├── metrics.py               # Performance metrics and safety limits
│   └── costs.py                 # Cost calculation utilities
├── data/                        # Data storage
│   ├── faq_document.txt         # Source HR FAQ document (1000+ words)
│   └── index/                   # Generated vector index files
├── outputs/                     # Output files
│   └── sample_queries.json      # Sample query-response pairs
├── tests/                       # Test suite
│   └── test_core.py             # Comprehensive tests
├── metrics/                     # Performance metrics
│   └── rag_metrics.csv          # Query performance logs
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

## Configuration

### Environment Variables

| Variable           | Default                | Description                                         |
| ------------------ | ---------------------- | --------------------------------------------------- |
| `OPENAI_API_KEY`   | Required               | Your OpenAI API key                                 |
| `EMBEDDING_MODEL`  | text-embedding-3-small | Embedding model to use                              |
| `GENERATION_MODEL` | gpt-4o-mini            | LLM model for answer generation                     |
| `CHUNK_SIZE`       | 500                    | Size of text chunks in characters                   |
| `CHUNK_OVERLAP`    | 100                    | Overlap between chunks                              |
| `SEARCH_K`         | 5                      | Number of chunks to retrieve                        |
| `SEARCH_TYPE`      | hybrid                 | Search strategy: 'hybrid', 'bm25', 'vector', 'auto' |
| `BM25_WEIGHT`      | 0.4                    | Weight for BM25 scores in hybrid search             |
| `VECTOR_WEIGHT`    | 0.6                    | Weight for vector scores in hybrid search           |
| `MAX_TOKENS`       | 8192                   | Maximum tokens for generation                       |
| `MAX_CHUNKS`       | 10                     | Maximum chunks per query                            |

### Safety Limits

The system includes built-in safety guardrails:

- **Token Limits**: Prevents requests exceeding model context windows
- **Chunk Limits**: Controls the number of chunks retrieved per query
- **Cost Limits**: Daily and per-query cost caps to prevent unexpected charges
- **Input Validation**: Validates all inputs and configurations

## Usage Examples

### Basic Query

```bash
python -m src.query "What is the probationary period for new employees?"
```

**Response Structure:**

```json
{
  "user_question": "What is the probationary period for new employees?",
  "system_answer": "The standard probationary period is 90 days for all full-time employees and 60 days for part-time employees...",
  "sources": [] // additional citing to catch hallucinations
  "chunks_related": [
    {
      "chunk_id": "qa_1",
      "chunk_type": "qa_pair",
      "similarity_score": 0.9234,
      "content_preview": "Q: How long is the probationary period...",
      "token_count": 67
    }
  ],
  "metadata": {
    "chunks_used": 2,
    "generation_tokens": { "input": 456, "output": 123 },
    "total_cost": 0.001846,
    "confidence": 0.8995,
    "latency": { "total_ms": 1272.3 }
  }
}
```

### Interactive Mode

```bash
python -m src.query --interactive
```

### System Statistics

```bash
python -m src.query --stats
```

### Quality Evaluation (0-10 Scoring)

```bash
# Evaluate complete RAG response with chunks
python -m src.evaluator --response-file outputs/sample_queries.json

# Evaluate specific question/answer pair
python -m src.evaluator --question "What is the vacation policy?" --answer "Employees get 15 days annually"

# Save evaluation to file
python -m src.evaluator --response-file response.json --output evaluation.json
```

**Evaluation Criteria:**

- **Relevance (0-3)**: How relevant are retrieved chunks to the question?
- **Accuracy (0-4)**: Is the answer factually correct based on context?
- **Completeness (0-2)**: Does the answer fully address the question?
- **Clarity (0-1)**: Is the answer clear and well-structured?

## Technical Implementation

### Chunking Strategy

The system uses a multi-strategy approach to text chunking:

1. **Q&A Structure Recognition**: Automatically detects FAQ format and preserves question-answer pairs
2. **Topic-Based Segmentation**: Groups content by major headings and topics
3. **Recursive Chunking**: Falls back to size-based chunking with smart splitting on sentence boundaries
4. **Overlap Management**: Maintains context continuity between adjacent chunks

**Why this approach?** FAQ documents have natural structure that should be preserved. The multi-strategy approach ensures we get both semantically coherent chunks and meet the minimum 20-chunk requirement.

### Hybrid Search Architecture

**Three-Component Search System:**

1. **BM25 Keyword Search**: Fast, exact keyword matching for specific terms and phrases
2. **Dense Vector Search**: OpenAI text-embedding-3-small (1536 dimensions) for semantic understanding
3. **Reciprocal Rank Fusion (RRF)**: Combines BM25 and vector results using weighted ranking

**Advanced Features:**

- **Adaptive Search Strategy**: Automatically selects optimal approach (keyword, semantic, or hybrid) based on query characteristics
- **Enhanced Metadata Filtering**: Topic categorization (benefits, payroll, time_off, etc.) for precise results
- **Transparent Scoring**: Returns BM25 scores, vector similarities, and RRF scores for full transparency

**Why this approach?** Research shows hybrid search outperforms individual methods by 15-25%. BM25 captures exact matches while vectors understand context. RRF provides optimal fusion of both approaches.

### RAG Architecture

1. **Retrieval Phase**: Query embedding → similarity search → relevance filtering
2. **Augmentation Phase**: Context building with chunk metadata and structure preservation
3. **Generation Phase**: Structured prompt with context → LLM generation → response formatting

**Why this design?** This clean separation allows for optimization of each component independently and provides transparency into the retrieval process.

## Performance Metrics

The system tracks comprehensive metrics for each query:

- **Latency**: Search time, generation time, total time
- **Costs**: Embedding costs, generation costs, total costs
- **Quality**: Chunk relevance, answer confidence, evaluation scores
- **Usage**: Token consumption, chunk utilization, daily usage

View metrics summary:

```bash
python -c "from src.metrics import MetricsCollector; print(MetricsCollector().get_summary_stats())"
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/test_core.py::TestTextChunker -v
```

## Known Limitations

1. **Context Window**: Limited by LLM context window (8K tokens for gpt-4o-mini)
2. **Embedding Costs**: Each query requires embedding generation (mitigated with caching strategies)
3. **Language Support**: Optimized for English text (FAQ document is in English)
4. **Real-time Updates**: Requires reindexing when source document changes
5. **Chunk Quality**: Automatic chunking may occasionally split related information

## Development

### Adding New Features

1. **New Chunking Strategy**: Extend `TextChunker` class in `src/chunking.py`
2. **Different Vector Store**: Implement new backend in `src/embeddings.py`
3. **Custom Evaluator**: Modify evaluation criteria in `src/evaluator.py`
4. **Additional Metrics**: Extend `MetricsCollector` in `src/metrics.py`

### Pre-commit Setup

```bash
pre-commit install
```

This project uses:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **gitleaks** for security scanning

## Production Considerations

### Scaling

- Implement vector database (Pinecone, Weaviate) for large document sets
- Add caching layer for frequent queries
- Consider batch processing for multiple queries

### Security

- API key rotation and secure storage
- Input sanitization and validation
- Audit logging for compliance

### Monitoring

- Production metrics dashboard
- Cost alerting and budgets
- Quality monitoring and A/B testing

## Cost Optimization

### Current Costs (Approximate)

- **Embedding**: ~\$0.00002 per query (text-embedding-3-small)
- **Generation**: ~\$0.002 per query (gpt-4o-mini, avg response)
- **Daily Limit**: \$50 (configurable)

### Optimization Strategies

1. **Embedding Caching**: Cache query embeddings for repeated questions and to save cost of calling the api for repeated questions
2. **Chunk Reranking**: Use cheaper reranking instead of larger embeddings
3. **Model Selection**: Balance cost vs. quality for your use case
4. **Batch Processing**: Group multiple queries when possible

## Support

For technical issues or questions:

1. Check the test suite for usage examples
2. Review the metrics logs for debugging information
3. Consult the source code documentation
4. Test with different configuration parameters

## Changelog

- Initial implementation with multi-strategy chunking
- Vector search with cosine similarity
- Cost tracking and safety limits
- Quality evaluation agent
- Comprehensive test suite and documentation

## Author

This code was written by Asoh Emmanuel Kaego (Devops/MLOps AI Engineer)
