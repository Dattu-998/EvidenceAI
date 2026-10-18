from pydantic import BaseModel, Field, ConfigDict
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
