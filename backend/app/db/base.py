"""
SQLAlchemy base configuration.
"""
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import Column, DateTime, MetaData, Uuid
from sqlalchemy.orm import DeclarativeBase, declared_attr
import uuid


# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    
    metadata = metadata
    
    # Generate __tablename__ automatically
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()
    
    # Common columns for all models
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    # timezone=True is required: the defaults below are tz-aware, and asyncpg
    # refuses to bind an aware datetime to a naive TIMESTAMP column.
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    
    def dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
