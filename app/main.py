from fastapi import FastAPI, Depends
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
