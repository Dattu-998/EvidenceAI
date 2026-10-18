from typing import List, Dict, Any
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
