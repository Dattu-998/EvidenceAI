from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.answer.refusal import evaluate_evidence_sufficiency
import logging

logger = logging.getLogger(__name__)

def generate_answer(query: str, evidence: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not evaluate_evidence_sufficiency(query, evidence):
        return "I could not find sufficient evidence to answer your question.", []

    citations = []
    for doc in evidence:
        citations.append({
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "page_number": doc["page_number"],
            "text_snippet": doc["text"][:100] + "..."
        })

    if settings.LLM_PROVIDER == "dummy":
        answer = f"Based on the provided evidence, here is a mock answer to '{query}'."
        return answer, citations

    elif settings.LLM_PROVIDER == "groq":
        try:
            from groq import Groq
            if not settings.GROQ_API_KEY:
                logger.error("GROQ_API_KEY is not set.")
                return "Error: Groq API key is missing. Please check your configuration.", citations

            client = Groq(api_key=settings.GROQ_API_KEY)

            context = ""
            for doc in evidence:
                context += f"- [Source: {doc['filename']}, Page {doc['page_number']}]: {doc['text']}\n"

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions strictly based on the provided evidence. Do not include outside knowledge. If the provided evidence is insufficient to answer the question, state that you cannot answer."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]

            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0,
                max_tokens=512,
                timeout=15.0
            )

            answer = chat_completion.choices[0].message.content
            return answer, citations

        except ImportError:
            logger.error("Groq SDK is not installed.")
            return "Error: Groq SDK not installed.", citations
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return "An error occurred while generating the answer from the LLM provider.", citations

    else:
        raise NotImplementedError(f"LLM Provider {settings.LLM_PROVIDER} is not fully implemented yet.")
