"""
Media item model for uploaded files.
"""
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship

from app.db.base import Base


class MediaItem(Base):
    """Uploaded media file model."""
    
    __tablename__ = "media_items"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # File info
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    sha256 = Column(String(64), unique=False, index=True, nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    
    # Media type
    media_type = Column(String(50), nullable=False)  # video, audio, image
    mime_type = Column(String(100), nullable=False)
    
    # Duration (for video/audio)
    duration_ms = Column(Integer, nullable=True)
    
    # Storage
    storage_path = Column(String(500), nullable=False)
    thumbnail_path = Column(String(500), nullable=True)
    
    # Metadata
    meta_info = Column(JSON, default={})
    
    # Retention
    expires_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        UniqueConstraint('sha256', 'user_id', name='uq_media_sha256_user'),
    )

    # Relationships
    user = relationship("User", back_populates="media_items")
    analysis_jobs = relationship("AnalysisJob", back_populates="media_item", cascade="all, delete-orphan")
