from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "EvidenceAI"
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/evidenceai"
    LLM_PROVIDER: str = "dummy"
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL_PATH: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
