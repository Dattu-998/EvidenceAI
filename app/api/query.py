from fastapi import APIRouter, Depends, HTTPException
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
        # Get more documents for hybrid before reranking
        evidence = hybrid_search(db, request.query, top_k=request.top_k * 2, document_ids=user_docs)
    else:
        raise HTTPException(status_code=400, detail="Invalid retrieval mode")

    from app.retrieval.reranker import rerank_documents
    evidence = rerank_documents(request.query, evidence, top_k=request.top_k)

    answer, citations = generate_answer(request.query, evidence)

    return QueryResponse(answer=answer, citations=citations)
