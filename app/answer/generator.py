from typing import List, Dict, Any, Tuple
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
