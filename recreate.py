import os

files = {
    "app/core/config.py": """from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "EvidenceAI"
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/evidenceai"
    LLM_PROVIDER: str = "dummy"
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL_PATH: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
""",
    "app/db/session.py": """from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
""",
    "app/db/models.py": """import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.db.session import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    owner_id = Column(String, nullable=False)
    status = Column(String, default="processing") # processing, ready, failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    error_message = Column(String, nullable=True)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    # 384 dimensions for all-MiniLM-L6-v2
    embedding = Column(Vector(384))

    document = relationship("Document", back_populates="chunks")
""",
    "app/ingestion/parser.py": """import io
import csv
import pdfplumber
from typing import List, Dict, Any, Union

def parse_pdf(file_content: bytes) -> List[Dict[str, Union[int, str]]]:
    pages_data = []
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_data.append({
                    "page_number": i + 1,
                    "text": text
                })
    return pages_data

def parse_csv(file_content: bytes) -> List[Dict[str, Union[int, str]]]:
    content_str = file_content.decode('utf-8')
    reader = csv.reader(io.StringIO(content_str))

    rows_data = []
    header = next(reader, None)

    for i, row in enumerate(reader):
        row_text = ""
        if header:
            row_text = ", ".join([f"{header[j]}: {val}" for j, val in enumerate(row) if j < len(header)])
        else:
            row_text = ", ".join(row)

        if row_text:
            rows_data.append({
                "page_number": i + 1,
                "text": row_text
            })

    return rows_data

def parse_file(filename: str, file_content: bytes) -> List[Dict[str, Union[int, str]]]:
    if filename.lower().endswith('.pdf'):
        return parse_pdf(file_content)
    elif filename.lower().endswith('.csv'):
        return parse_csv(file_content)
    else:
        raise ValueError(f"Unsupported file format for {filename}")
""",
    "app/ingestion/chunker.py": """from typing import List, Dict, Union

def chunk_text(pages_data: List[Dict[str, Union[int, str]]], chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Union[int, str]]]:
    chunks = []
    chunk_index = 0

    for page in pages_data:
        page_num = page["page_number"]
        text = page["text"]

        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start + chunk_size // 2:
                    end = last_space

            chunk_text_content = text[start:end].strip()

            if chunk_text_content:
                chunks.append({
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "text": chunk_text_content
                })
                chunk_index += 1

            start = end - overlap if end < text_len else text_len

    return chunks
""",
    "app/retrieval/vector.py": """from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db.models import DocumentChunk
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
    return _model

def get_embedding(text: str) -> List[float]:
    model = get_model()
    return model.encode(text).tolist()

def search_vectors(db: Session, query: str, top_k: int = 5, document_ids: List[str] = None) -> List[Dict[str, Any]]:
    query_embedding = get_embedding(query)

    stmt = select(DocumentChunk).order_by(
        DocumentChunk.embedding.cosine_distance(query_embedding)
    ).limit(top_k)

    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

    results = db.execute(stmt).scalars().all()

    return [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "filename": chunk.document.filename if chunk.document else "Unknown"
        }
        for chunk in results
    ]
""",
    "app/retrieval/bm25.py": """from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db.models import DocumentChunk
from rank_bm25 import BM25Okapi

def search_bm25(db: Session, query: str, top_k: int = 5, document_ids: List[str] = None) -> List[Dict[str, Any]]:
    stmt = select(DocumentChunk)
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

    all_chunks = db.execute(stmt).scalars().all()

    if not all_chunks:
        return []

    tokenized_corpus = [chunk.text.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)

    scored_chunks = sorted(zip(all_chunks, doc_scores), key=lambda x: x[1], reverse=True)

    top_results = [chunk for chunk, score in scored_chunks[:top_k] if score > 0]

    return [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "filename": chunk.document.filename if chunk.document else "Unknown"
        }
        for chunk in top_results
    ]
""",
    "app/retrieval/hybrid.py": """from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.retrieval.vector import search_vectors
from app.retrieval.bm25 import search_bm25

def hybrid_search(db: Session, query: str, top_k: int = 5, document_ids: List[str] = None) -> List[Dict[str, Any]]:
    vector_results = search_vectors(db, query, top_k=top_k * 2, document_ids=document_ids)
    bm25_results = search_bm25(db, query, top_k=top_k * 2, document_ids=document_ids)

    rrf_score = {}
    k = 60

    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        if doc_id not in rrf_score:
            rrf_score[doc_id] = {"score": 0, "doc": doc}
        rrf_score[doc_id]["score"] += 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        if doc_id not in rrf_score:
            rrf_score[doc_id] = {"score": 0, "doc": doc}
        rrf_score[doc_id]["score"] += 1 / (k + rank + 1)

    sorted_results = sorted(rrf_score.values(), key=lambda x: x["score"], reverse=True)

    return [item["doc"] for item in sorted_results[:top_k]]
""",
    "app/answer/refusal.py": """from typing import List, Dict, Any

def evaluate_evidence_sufficiency(query: str, evidence: List[Dict[str, Any]]) -> bool:
    if not evidence:
        return False

    combined_evidence = " ".join([e["text"] for e in evidence])
    if len(combined_evidence.strip()) < 50:
        return False

    return True
""",
    "app/answer/generator.py": """from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.answer.refusal import evaluate_evidence_sufficiency

def generate_answer(query: str, evidence: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not evaluate_evidence_sufficiency(query, evidence):
        return "I could not find sufficient evidence to answer your question.", []

    if settings.LLM_PROVIDER == "dummy":
        answer = f"Based on the provided evidence, here is a mock answer to '{query}'."
        citations = []
        for doc in evidence:
            citations.append({
                "document_id": doc["document_id"],
                "filename": doc["filename"],
                "page_number": doc["page_number"],
                "text_snippet": doc["text"][:100] + "..."
            })
        return answer, citations
    else:
        raise NotImplementedError(f"LLM Provider {settings.LLM_PROVIDER} is not fully implemented yet.")
""",
    "app/schemas.py": """from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any
from datetime import datetime

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    created_at: datetime
    error_message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class QueryRequest(BaseModel):
    query: str
    retrieval_mode: str = Field(default="hybrid", description="vector, bm25, or hybrid")
    top_k: int = 5

class Citation(BaseModel):
    document_id: str
    filename: str
    page_number: Optional[int] = None
    text_snippet: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
""",
    "app/security/access.py": """def get_current_user_id() -> str:
    return "dummy_user_123"
""",
    "app/api/documents.py": """from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import List
from app.db.session import get_db, SessionLocal
from app.db.models import Document, DocumentChunk
from app.schemas import DocumentResponse
from app.security.access import get_current_user_id
from app.ingestion.parser import parse_file
from app.ingestion.chunker import chunk_text
from app.retrieval.vector import get_embedding
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

def process_document(document_id: str, file_content: bytes, filename: str):
    with SessionLocal() as db:
        doc = db.get(Document, document_id)
        if not doc:
            return

        try:
            pages_data = parse_file(filename, file_content)
            chunks = chunk_text(pages_data)

            db_chunks = []
            for chunk in chunks:
                embedding = get_embedding(chunk["text"])
                db_chunk = DocumentChunk(
                    document_id=document_id,
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    embedding=embedding
                )
                db_chunks.append(db_chunk)

            db.add_all(db_chunks)
            doc.status = "ready"
            db.commit()
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            doc.status = "failed"
            doc.error_message = str(e)
            db.commit()

@router.post("/documents", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    if not file.filename.lower().endswith(('.pdf', '.csv')):
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
         raise HTTPException(status_code=400, detail="File too large")

    existing_doc = db.execute(
        select(Document).where(Document.owner_id == user_id, Document.filename == file.filename)
    ).scalar_one_or_none()

    if existing_doc:
        raise HTTPException(status_code=409, detail="Document with this name already exists")

    doc = Document(filename=file.filename, owner_id=user_id, status="processing")
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(process_document, doc.id, content, file.filename)

    return doc

@router.get("/documents", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    docs = db.execute(select(Document).where(Document.owner_id == user_id)).scalars().all()
    return docs

@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    doc = db.get(Document, document_id)
    if not doc or doc.owner_id != user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    doc = db.get(Document, document_id)
    if not doc or doc.owner_id != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()
    return {"message": "Document deleted"}
""",
    "app/api/query.py": """from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db.session import get_db
from app.db.models import Document
from app.schemas import QueryRequest, QueryResponse
from app.security.access import get_current_user_id
from app.retrieval.vector import search_vectors
from app.retrieval.bm25 import search_bm25
from app.retrieval.hybrid import hybrid_search
from app.answer.generator import generate_answer

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    user_docs = db.execute(
        select(Document.id).where(Document.owner_id == user_id, Document.status == "ready")
    ).scalars().all()

    if not user_docs:
        raise HTTPException(status_code=400, detail="No processed documents available for query.")

    evidence = []
    if request.retrieval_mode == "vector":
        evidence = search_vectors(db, request.query, top_k=request.top_k, document_ids=user_docs)
    elif request.retrieval_mode == "bm25":
        evidence = search_bm25(db, request.query, top_k=request.top_k, document_ids=user_docs)
    elif request.retrieval_mode == "hybrid":
        evidence = hybrid_search(db, request.query, top_k=request.top_k, document_ids=user_docs)
    else:
        raise HTTPException(status_code=400, detail="Invalid retrieval mode")

    answer, citations = generate_answer(request.query, evidence)

    return QueryResponse(answer=answer, citations=citations)
""",
    "app/main.py": """from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.documents import router as documents_router
from app.api.query import router as query_router
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="EvidenceAI")

app.include_router(documents_router, tags=["Documents"])
app.include_router(query_router, tags=["Query"])

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/ready")
def ready_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}
""",
    "tests/test_api.py": """import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import io
import time

from app.main import app
from app.db.session import Base, get_db
from app.db.models import Document, DocumentChunk
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    db.query(DocumentChunk).delete()
    db.query(Document).delete()
    db.commit()
    db.close()
    yield
    db = TestingSessionLocal()
    db.query(DocumentChunk).delete()
    db.query(Document).delete()
    db.commit()
    db.close()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_upload_and_query_pdf():
    from reportlab.pdfgen import canvas
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 750, "EvidenceAI is an advanced document intelligence platform.")
    c.drawString(100, 730, "It uses pgvector for similarity search.")
    c.showPage()
    c.save()
    pdf_bytes = pdf_buffer.getvalue()

    response = client.post(
        "/documents",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    doc = response.json()
    assert doc["filename"] == "test.pdf"
    assert doc["status"] == "processing"

    time.sleep(3)

    response = client.get(f"/documents/{doc['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

    query_req = {
        "query": "What is EvidenceAI?",
        "retrieval_mode": "hybrid",
        "top_k": 2
    }
    response = client.post("/query", json=query_req)
    assert response.status_code == 200
    result = response.json()

    assert result["answer"]
    assert len(result["citations"]) > 0
    assert result["citations"][0]["filename"] == "test.pdf"

def test_unauthorized_access():
    response = client.get("/documents/non-existent-id")
    assert response.status_code == 404
""",
    "tests/test_ingestion.py": """import pytest
from app.ingestion.parser import parse_csv, parse_pdf
from app.ingestion.chunker import chunk_text
import io
from reportlab.pdfgen import canvas

def test_parse_csv():
    csv_content = b"header1,header2\\nval1,val2\\nval3,val4"
    data = parse_csv(csv_content)
    assert len(data) == 2
    assert data[0]["text"] == "header1: val1, header2: val2"
    assert data[0]["page_number"] == 1
    assert data[1]["text"] == "header1: val3, header2: val4"
    assert data[1]["page_number"] == 2

def test_parse_pdf():
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 750, "Page 1 content")
    c.showPage()
    c.drawString(100, 750, "Page 2 content")
    c.showPage()
    c.save()

    data = parse_pdf(pdf_buffer.getvalue())
    assert len(data) == 2
    assert "Page 1 content" in data[0]["text"]
    assert data[0]["page_number"] == 1
    assert "Page 2 content" in data[1]["text"]
    assert data[1]["page_number"] == 2

def test_chunk_text():
    pages_data = [
        {"page_number": 1, "text": "A" * 1500}
    ]
    chunks = chunk_text(pages_data, chunk_size=1000, overlap=100)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert len(chunks[0]["text"]) == 1000
    assert chunks[1]["page_number"] == 1
    assert len(chunks[1]["text"]) == 600
""",
    "tests/test_retrieval.py": """import pytest
from app.retrieval.vector import get_embedding

def test_get_embedding():
    text = "This is a test document."
    embedding = get_embedding(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert isinstance(embedding[0], float)
""",
    "requirements.txt": """fastapi>=0.111.0
uvicorn>=0.30.0
pydantic>=2.7.0
pydantic-settings>=2.2.0
SQLAlchemy>=2.0.30
alembic>=1.13.1
psycopg-binary>=3.1.18
pgvector>=0.2.5
python-multipart>=0.0.9
pdfplumber>=0.11.0
sentence-transformers>=2.7.0
rank_bm25>=0.2.2
httpx>=0.27.0
pytest>=8.2.0
pytest-asyncio>=0.23.6
python-dotenv>=1.0.1
reportlab>=4.0.0
""",
    "Dockerfile": """FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "docker-compose.yml": """version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/evidenceai
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - .:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=evidenceai
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d evidenceai"]
      interval: 5s
      timeout: 5s
      retries: 5
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
""",
    ".env.example": """DATABASE_URL=postgresql://user:password@localhost:5432/evidenceai
LLM_PROVIDER=dummy
OPENAI_API_KEY=
EMBEDDING_MODEL_PATH=all-MiniLM-L6-v2
""",
    ".gitignore": """__pycache__/
*.pyc
.env
.venv
venv/
env/
.pytest_cache/
htmlcov/
.coverage
data/
*.db
*.sqlite3
""",
    "README.md": """# EvidenceAI

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
- `LLM_PROVIDER`: Choose `dummy` for mock testing or future real providers.
- `EMBEDDING_MODEL_PATH`: Change the sentence-transformer model used (defaults to `all-MiniLM-L6-v2`).

## API Examples

### Upload Document
```bash
curl -X POST "http://localhost:8000/documents" \\
     -H "accept: application/json" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@sample.pdf"
```

### Query
```bash
curl -X POST "http://localhost:8000/query" \\
     -H "accept: application/json" \\
     -H "Content-Type: application/json" \\
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
- The LLM Provider logic currently utilizes a `dummy` generator. Real provider integrations (OpenAI, Ollama, etc.) are stubbed and marked as Phase D extensions.
- Multi-user authentication is currently mocked to use a static test user in the `app.security.access` module.
- BM25 retrieval uses an in-memory Okapi BM25 implementation which loads all document chunks for scoring. It will not scale for huge text bases but suffices for the baseline slice.
"""
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
