"""
Analysis job and related models.
"""
from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship

from app.db.base import Base


class AnalysisJob(Base):
    """Analysis job model tracking the processing state."""
    
    __tablename__ = "analysis_jobs"
    
    media_id = Column(UUID(as_uuid=True), ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Status
    status = Column(String(50), default="PENDING", nullable=False, index=True)
    stage = Column(String(50), default="PENDING", nullable=False)
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    
    # Options
    options = Column(JSON, default={})
    
    # Results
    results = Column(JSON, nullable=True)
    overall_score = Column(Float, nullable=True)
    label = Column(String(50), nullable=True)  # AUTHENTIC, LIKELY_FAKE, FAKE
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Celery task ID
    celery_task_id = Column(String(255), nullable=True)
    
    # Relationships
    media_item = relationship("MediaItem", back_populates="analysis_jobs")
    segments = relationship("Segment", back_populates="analysis_job", cascade="all, delete-orphan")
    model_runs = relationship("ModelRun", back_populates="analysis_job", cascade="all, delete-orphan")
    evidence_artifacts = relationship("EvidenceArtifact", back_populates="analysis_job", cascade="all, delete-orphan")
    report = relationship("Report", back_populates="analysis_job", uselist=False, cascade="all, delete-orphan")


class Segment(Base):
    """Flagged segment within analyzed media."""
    
    __tablename__ = "segments"
    
    job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Time range
    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)
    
    # Detection info
    segment_type = Column(String(50), nullable=False)  # video, audio, lipsync
    score = Column(Float, nullable=False)
    reason = Column(String(255), nullable=False)
    
    # Additional data
    meta_info = Column(JSON, default={})
    
    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="segments")


class ModelRun(Base):
    """Record of a model inference run."""
    
    __tablename__ = "model_runs"
    
    job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Model info
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Results
    predictions = Column(JSON, nullable=True)
    score = Column(Float, nullable=True)
    
    # Performance
    inference_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="model_runs")
