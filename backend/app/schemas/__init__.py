"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field


# ============ Auth Schemas ============

class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response."""
    id: UUID
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# ============ Media Schemas ============

class MediaUploadResponse(BaseModel):
    """Media upload response."""
    id: UUID
    filename: str
    sha256: str
    media_type: str
    duration_ms: Optional[int]
    file_size: int
    created_at: datetime

    class Config:
        from_attributes = True


class MediaItemResponse(BaseModel):
    """Media item detail response."""
    id: UUID
    filename: str
    original_filename: str
    sha256: str
    media_type: str
    mime_type: str
    duration_ms: Optional[int]
    file_size: int
    thumbnail_path: Optional[str]
    metadata: dict
    created_at: datetime
    expires_at: Optional[datetime]

    class Config:
        from_attributes = True


# ============ Analysis Schemas ============

class AnalysisOptions(BaseModel):
    """Analysis job options."""
    run_video: bool = True
    run_audio: bool = True
    run_lipsync: bool = True
    run_identity: bool = False
    privacy_mode: bool = False


class AnalysisStartRequest(BaseModel):
    """Start analysis request."""
    media_id: UUID
    options: Optional[AnalysisOptions] = None


class AnalysisStartResponse(BaseModel):
    """Start analysis response."""
    job_id: UUID
    status: str
    stage: str


class AnalysisStatusResponse(BaseModel):
    """Analysis job status response."""
    job_id: UUID
    status: str
    stage: str
    progress: float
    error_message: Optional[str]
    started_at: Optional[datetime]


class SegmentResponse(BaseModel):
    """Flagged segment response."""
    id: UUID
    start_ms: int
    end_ms: int
    segment_type: str
    score: float
    reason: str

    class Config:
        from_attributes = True


class ModelRunResponse(BaseModel):
    """Model run response."""
    model_name: str
    model_version: str
    score: Optional[float]
    inference_time_ms: Optional[int]

    class Config:
        from_attributes = True


class AnalysisResultResponse(BaseModel):
    """Full analysis results response."""
    job_id: UUID
    status: str
    overall_score: Optional[float]
    label: Optional[str]
    results: Optional[dict] = None  # Contains all detailed forensic data
    segments: List[SegmentResponse] = []
    model_runs: List[ModelRunResponse] = []
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


# ============ Evidence Schemas ============

class TimelineMarker(BaseModel):
    """Timeline marker for quick rendering."""
    timestamp_ms: int
    segment_type: str
    score: float
    reason: str


class TimelineResponse(BaseModel):
    """Evidence timeline response."""
    job_id: UUID
    duration_ms: Optional[int]
    markers: List[TimelineMarker] = []


class EvidenceArtifactResponse(BaseModel):
    """Evidence artifact response."""
    id: UUID
    artifact_type: str
    timestamp_ms: Optional[str]
    storage_path: str

    class Config:
        from_attributes = True


class TranscriptWord(BaseModel):
    """Transcript word with timing."""
    word: str
    start_ms: int
    end_ms: int
    confidence: float


class TranscriptResponse(BaseModel):
    """Transcript response."""
    job_id: UUID
    words: List[TranscriptWord] = []
    full_text: str = ""


# ============ Report Schemas ============

class ReportGenerateRequest(BaseModel):
    """Report generation request."""
    style: str = "detailed"  # detailed, summary, executive


class ReportResponse(BaseModel):
    """Report response."""
    job_id: UUID
    id: UUID
    created_at: datetime
    overall_score: Optional[float] = 0.0
    media_type: Optional[str] = "unknown"
    summary: Optional[str]
    full_report: Optional[dict]
    generated_at: Optional[datetime]

    class Config:
        from_attributes = True


# ============ Admin Schemas ============

class BatchScanRequest(BaseModel):
    """Batch scan request."""
    media_ids: List[UUID]
    options: Optional[AnalysisOptions] = None


class BatchScanResponse(BaseModel):
    """Batch scan response."""
    job_ids: List[UUID]
    total: int


class QueueStatusResponse(BaseModel):
    """Queue status response."""
    pending: int
    processing: int
    completed: int
    failed: int


class LabelFeedbackRequest(BaseModel):
    """Active learning label feedback."""
    job_id: UUID
    correct_label: str
    notes: Optional[str] = None
