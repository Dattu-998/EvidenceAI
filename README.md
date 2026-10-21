# EvidenceAI

Evaluated Document Intelligence Platform answering questions over PDFs and CSVs using verifiable evidence, page-level citations, hybrid retrieval, reranking, and evidence-based refusal.

## Setup Instructions

### Environment
1. Clone the repository
2. Ensure you have Python 3.12+ and Docker installed.
3. Copy the example `.env` file:
   ```bash
   cp .env.example .env
   ```

### Running with Docker

Run the PostgreSQL instance with pgvector via Docker Compose:
```bash
docker compose up -d db
```

Run migrations (if you have local env setup):
```bash
alembic upgrade head
```

Run the API:
```bash
docker compose up api
```
*(Or use `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` locally)*

### Local Development Setup

To run locally without the API docker container:
1. Create a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the PostgreSQL pgvector DB in docker, then run `alembic upgrade head`.

## Architecture
- **FastAPI**: Provides a fast and asynchronous backend layer.
- **PostgreSQL & pgvector**: Used for structured relational data and vector similarity search.
- **SQLAlchemy & Alembic**: Manages database models and schema migrations.
- **sentence-transformers**: Provides embedding capabilities using the `all-MiniLM-L6-v2` model.
- **BM25**: Used for lexical retrieval baseline.
- **Hybrid Retrieval**: Combines BM25 and Vector search results using Reciprocal Rank Fusion (RRF).

## Configuration
See `.env.example` for all configurable environment variables.
- `DATABASE_URL`: PostgreSQL connection string.
- `LLM_PROVIDER`: Choose `dummy` for mock testing, or `groq` to use the Groq API for generation.
- `GROQ_API_KEY`: Required if `LLM_PROVIDER` is set to `groq`. Paste your key here to connect to Groq.
- `EMBEDDING_MODEL_PATH`: Change the sentence-transformer model used (defaults to `all-MiniLM-L6-v2`).

## API Examples

### Upload Document
```bash
curl -X POST "http://localhost:8000/documents" \
     -H "accept: application/json" \
     -H "Authorization: Bearer test_user_1" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.pdf"
```

### Query
```bash
curl -X POST "http://localhost:8000/query" \
     -H "accept: application/json" \
     -H "Authorization: Bearer test_user_1" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is the main topic of the document?",
           "retrieval_mode": "hybrid",
           "top_k": 5
         }'
```

## Testing
Unit and integration tests are handled using `pytest`. The tests will require a running pgvector instance.

Run tests:
```bash
PYTHONPATH=. pytest
```

## Known Limitations
- The LLM Provider logic supports a `dummy` generator for offline testing, and a `groq` provider using `qwen/qwen3.8-27b`. Other integrations (OpenAI, local Ollama) remain as future extensions.
- Multi-user authentication is currently simulated using the `Authorization: Bearer <user_id>` header. A real JWT token verifier would be needed for production.
- BM25 retrieval uses an in-memory Okapi BM25 implementation which loads all document chunks for scoring. It will not scale for huge text bases but suffices for the baseline slice.
