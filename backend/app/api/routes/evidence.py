"""
Evidence artifact routes.
"""
from uuid import UUID
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db import get_async_db
from app.models import User, MediaItem, AnalysisJob, Segment, EvidenceArtifact
from app.schemas import (
    TimelineResponse,
    TimelineMarker,
    EvidenceArtifactResponse,
    TranscriptResponse,
)
from app.api.deps import get_current_user


router = APIRouter(prefix="/analysis", tags=["Evidence"])


@router.get("/{job_id}/evidence/timeline", response_model=TimelineResponse)
async def get_evidence_timeline(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get timeline markers for evidence visualization."""
    result = await db.execute(
        select(AnalysisJob)
        .options(
            selectinload(AnalysisJob.segments),
            selectinload(AnalysisJob.media_item)
        )
        .join(MediaItem)
        .where(
            AnalysisJob.id == job_id,
            MediaItem.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
    
    markers = [
        TimelineMarker(
            timestamp_ms=seg.start_ms,
            segment_type=seg.segment_type,
            score=seg.score,
            reason=seg.reason
        )
        for seg in sorted(job.segments, key=lambda x: x.start_ms)
    ]
    
    return TimelineResponse(
        job_id=job.id,
        duration_ms=job.media_item.duration_ms if job.media_item else None,
        markers=markers
    )


@router.get("/{job_id}/evidence/frame/{frame_id}")
async def get_evidence_frame(
    job_id: UUID,
    frame_id: str,
    heatmap: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific frame, optionally with heatmap overlay."""
    result = await db.execute(
        select(EvidenceArtifact)
        .join(AnalysisJob)
        .join(MediaItem)
        .where(
            AnalysisJob.id == job_id,
            MediaItem.user_id == current_user.id,
            EvidenceArtifact.id == frame_id
        )
    )
    artifact = result.scalar_one_or_none()
    
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frame not found"
        )
    
    file_path = Path(artifact.storage_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frame file not found"
        )
    
    return FileResponse(
        path=str(file_path),
        media_type="image/png"
    )


@router.get("/{job_id}/evidence/spectrogram/{segment_id}")
async def get_evidence_spectrogram(
    job_id: UUID,
    segment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get spectrogram image for an audio segment."""
    result = await db.execute(
        select(EvidenceArtifact)
        .join(AnalysisJob)
        .join(MediaItem)
        .where(
            AnalysisJob.id == job_id,
            MediaItem.user_id == current_user.id,
            EvidenceArtifact.id == segment_id,
            EvidenceArtifact.artifact_type == "spectrogram"
        )
    )
    artifact = result.scalar_one_or_none()
    
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Spectrogram not found"
        )
    
    file_path = Path(artifact.storage_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Spectrogram file not found"
        )
    
    return FileResponse(
        path=str(file_path),
        media_type="image/png"
    )


@router.get("/{job_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get timestamped transcript with confidence scores."""
    result = await db.execute(
        select(AnalysisJob)
        .join(MediaItem)
        .where(
            AnalysisJob.id == job_id,
            MediaItem.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
    
    # Get transcript from job results
    transcript_data = job.results.get("transcript", {}) if job.results else {}
    
    return TranscriptResponse(
        job_id=job.id,
        words=transcript_data.get("words", []),
        full_text=transcript_data.get("full_text", "")
    )
