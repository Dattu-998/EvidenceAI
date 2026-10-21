from typing import List, Dict, Any
from app.core.config import settings
from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None and settings.RERANKER_MODEL_PATH:
        try:
            _reranker = CrossEncoder(settings.RERANKER_MODEL_PATH)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to load reranker: {e}")
    return _reranker

def rerank_documents(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Reranks documents using a CrossEncoder.
    If reranker is not configured, returns the original list up to top_k.
    """
    reranker = get_reranker()
    if not reranker or not documents:
        return documents[:top_k]

    pairs = [[query, doc["text"]] for doc in documents]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for i, doc in enumerate(documents):
        doc["_rerank_score"] = float(scores[i])

    reranked = sorted(documents, key=lambda x: x["_rerank_score"], reverse=True)
    return reranked[:top_k]
