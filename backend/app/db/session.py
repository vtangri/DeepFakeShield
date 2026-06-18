"""
Database session management.
"""
from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings



# Sync engine for Alembic migrations
print(f"DEBUG: settings.DATABASE_URL={settings.DATABASE_URL}")
db_url = settings.DATABASE_URL
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://")

sync_engine = create_engine(
    db_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Async engine for FastAPI
async_db_url = settings.DATABASE_URL
if async_db_url.startswith("postgresql://"):
    async_db_url = async_db_url.replace("postgresql://", "postgresql+asyncpg://")
elif async_db_url.startswith("sqlite://"):
    async_db_url = async_db_url.replace("sqlite://", "sqlite+aiosqlite://")

async_engine = create_async_engine(
    async_db_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=settings.DEBUG,
)

# Session factories
SessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Session:
    """Get sync database session (for Celery workers)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session (for FastAPI)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
