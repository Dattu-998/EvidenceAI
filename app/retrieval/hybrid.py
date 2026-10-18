from typing import List, Dict, Any
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
