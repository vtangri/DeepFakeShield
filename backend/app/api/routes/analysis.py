"""
Analysis job routes.
"""
from datetime import datetime
from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db import get_async_db
from app.models import User, MediaItem, AnalysisJob, Segment, ModelRun
from app.core import TaskState
from app.schemas import (
    AnalysisStartRequest,
    AnalysisStartResponse,
    AnalysisStatusResponse,
    AnalysisResultResponse,
    SegmentResponse,
    ModelRunResponse,
)
from app.api.deps import get_current_user
from fastapi import BackgroundTasks
from app.services.simulation import simulate_analysis_pipeline


router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/start", response_model=AnalysisStartResponse, status_code=status.HTTP_201_CREATED)
async def start_analysis(
    request: AnalysisStartRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Start a new analysis job for a media item."""
    # Find the media item and verify ownership
    result = await db.execute(
        select(MediaItem).where(
            MediaItem.id == request.media_id,
            MediaItem.user_id == current_user.id
        )
    )
    media_item = result.scalar_one_or_none()
    
    if not media_item:
        print(f"DEBUG: Media item not found or unauthorized for id={request.media_id}, user_id={current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media item not found"
        )
    
    # Check for existing pending/running job
    result = await db.execute(
        select(AnalysisJob).where(
            AnalysisJob.media_id == request.media_id,
            AnalysisJob.status.in_([TaskState.PENDING, TaskState.VALIDATING, 
                                     TaskState.EXTRACTING, TaskState.TRANSCRIBING,
                                     TaskState.INFER_VIDEO, TaskState.INFER_AUDIO,
                                     TaskState.LIPSYNC, TaskState.FUSION, TaskState.REPORT])
        )
    )
    existing_job = result.scalar_one_or_none()
    
    if existing_job:
        return AnalysisStartResponse(
            job_id=existing_job.id,
            status=existing_job.status,
            stage=existing_job.stage
        )
    
    # Create new analysis job
    options = request.options.model_dump() if request.options else {}
    
    job = AnalysisJob(
        media_id=request.media_id,
        status=TaskState.PENDING,
        stage=TaskState.PENDING,
        options=options,
        started_at=datetime.utcnow()
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Trigger Simulation Task (Temporary until Celery is fully configured)
    background_tasks.add_task(simulate_analysis_pipeline, job.id)
    
    # from app.workers.preprocess import run_analysis_pipeline
    # run_analysis_pipeline.delay(str(job.id))
    
    return AnalysisStartResponse(
        job_id=job.id,
        status=job.status,
        stage=job.stage
    )


@router.get("/{job_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get the current status of an analysis job."""
    # Find by job ID and verify ownership through MediaItem
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
        print(f"DEBUG: Job not found with id={job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
    
    print(f"DEBUG: Found job id={job_id}, stage={job.stage}, status={job.status}")
    
    return AnalysisStatusResponse(
        job_id=job.id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        error_message=job.error_message,
        started_at=job.started_at
    )


@router.get("/{job_id}/result", response_model=AnalysisResultResponse)
async def get_analysis_result(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get the full results of a completed analysis job."""
    # Find by job ID and verify ownership through MediaItem
    result = await db.execute(
        select(AnalysisJob)
        .options(
            selectinload(AnalysisJob.segments),
            selectinload(AnalysisJob.model_runs)
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
    
    segments = [
        SegmentResponse(
            id=seg.id,
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            segment_type=seg.segment_type,
            score=seg.score,
            reason=seg.reason
        )
        for seg in job.segments
    ]
    
    model_runs = [
        ModelRunResponse(
            model_name=run.model_name,
            model_version=run.model_version,
            score=run.score,
            inference_time_ms=run.inference_time_ms
        )
        for run in job.model_runs
    ]
    
    return AnalysisResultResponse(
        job_id=job.id,
        status=job.status,
        overall_score=job.overall_score,
        label=job.label,
        results=job.results,  # Include detailed forensic data
        segments=segments,
        model_runs=model_runs,
        started_at=job.started_at,
        completed_at=job.completed_at
    )
