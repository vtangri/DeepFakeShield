"""
Application configuration using Pydantic Settings.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "DeepFakeShield AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://deepfakeshield:deepfakeshield@localhost:5432/deepfakeshield"
    )
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    
    # Security
    SECRET_KEY: str = Field(default="change-this-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Storage
    STORAGE_PATH: str = "./storage"
    UPLOAD_MAX_SIZE_MB: int = 500
    
    # Celery
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0")
    
    # ML Models
    ML_MODELS_PATH: str = "../ml/models"
    VIDEO_MODEL_VERSION: str = "v1.0.0"
    AUDIO_MODEL_VERSION: str = "v1.0.0"
    FUSION_MODEL_VERSION: str = "v1.0.0"
    
    # LLM
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4"
    
    # Feature Flags
    ENABLE_GPU: bool = False
    ENABLE_BATCH_PROCESSING: bool = True
    ENABLE_ACTIVE_LEARNING: bool = False
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
