"""
User model for authentication.
"""
from sqlalchemy import Column, String, Boolean
from sqlalchemy.orm import relationship

from app.db.base import Base


class User(Base):
    """User account model."""
    
    __tablename__ = "users"
    
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Relationships
    media_items = relationship("MediaItem", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
