"""
Evidence and Report models.
"""
from sqlalchemy import Column, String, ForeignKey, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship

from app.db.base import Base


class EvidenceArtifact(Base):
    """Generated evidence artifact (heatmap, spectrogram, etc.)."""
    
    __tablename__ = "evidence_artifacts"
    
    job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Artifact type
    artifact_type = Column(String(50), nullable=False)  # heatmap, spectrogram, mouth_roi, frame
    
    # Time reference (optional)
    timestamp_ms = Column(String(50), nullable=True)
    frame_number = Column(String(50), nullable=True)
    
    # Storage
    storage_path = Column(String(500), nullable=False)
    
    # Metadata
    meta_info = Column(JSON, default={})
    
    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="evidence_artifacts")


class Report(Base):
    """Generated forensic report."""
    
    __tablename__ = "reports"
    
    job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    
    # Report content
    summary = Column(Text, nullable=True)
    full_report = Column(JSON, nullable=True)
    
    # Export paths
    pdf_path = Column(String(500), nullable=True)
    json_path = Column(String(500), nullable=True)
    
    # LLM info
    llm_model_used = Column(String(100), nullable=True)
    
    # Generation time
    generated_at = Column(DateTime, nullable=True)
    
    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="report")


class AuditLog(Base):
    """Audit log for user actions."""
    
    __tablename__ = "audit_logs"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Action info
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Details
    details = Column(JSON, default={})
    
    # IP info
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
