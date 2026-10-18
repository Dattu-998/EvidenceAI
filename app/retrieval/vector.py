from typing import List, Dict, Any
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
