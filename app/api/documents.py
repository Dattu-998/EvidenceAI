from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
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
