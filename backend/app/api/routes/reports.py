"""
Report generation and export routes.
"""
from uuid import UUID
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db import get_async_db
from app.models import User, MediaItem, AnalysisJob, Report
from app.schemas import ReportGenerateRequest, ReportResponse
from app.api.deps import get_current_user


router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post("/{job_id}/report/generate", response_model=ReportResponse)
async def generate_report(
    job_id: UUID,
    request: ReportGenerateRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Generate or regenerate a forensic report for an analysis job."""
    result = await db.execute(
        select(AnalysisJob)
        .join(MediaItem, AnalysisJob.media_id == MediaItem.id)
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
    
    if job.status != "DONE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis must be completed before generating report"
        )
    
    # Check for existing report
    result = await db.execute(
        select(Report).where(Report.job_id == job_id)
    )
    report = result.scalar_one_or_none()
    
    if not report:
        # Create new report
        report = Report(
            job_id=job_id,
            generated_at=datetime.utcnow()
        )
        db.add(report)
    
    # TODO: Call LLM service to generate report
    # For now, generate a placeholder report
    report.summary = _generate_placeholder_summary(job)
    report.full_report = _generate_placeholder_full_report(job)
    report.generated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(report)
    
    return ReportResponse(
        id=report.id,
        job_id=job_id,
        summary=report.summary,
        full_report=report.full_report,
        generated_at=report.generated_at,
        created_at=report.created_at,
        overall_score=job.overall_score,
        media_type=job.media_item.media_type if hasattr(job, "media_item") and job.media_item else "unknown"
    )


@router.get("/{job_id}/report", response_model=ReportResponse)
async def get_report(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get the forensic report for an analysis job."""
    result = await db.execute(
        select(Report)
        .options(selectinload(Report.analysis_job).selectinload(AnalysisJob.media_item))
        .join(AnalysisJob)
        .join(MediaItem)
        .where(
            Report.job_id == job_id,
            MediaItem.user_id == current_user.id
        )
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found. Generate one first."
        )
    
    return ReportResponse(
        id=report.id,
        job_id=job_id,
        summary=report.summary,
        full_report=report.full_report,
        generated_at=report.generated_at,
        created_at=report.created_at,
        overall_score=report.analysis_job.overall_score if report.analysis_job else 0.0,
        media_type=report.analysis_job.media_item.media_type if report.analysis_job and report.analysis_job.media_item else "unknown"
    )


@router.get("/{job_id}/report.pdf")
async def get_report_pdf(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Download the report as PDF."""
    from app.services.pdf_service import generate_pdf_report
    
    result = await db.execute(
        select(Report)
        .options(selectinload(Report.analysis_job).selectinload(AnalysisJob.media_item))
        .join(AnalysisJob)
        .join(MediaItem)
        .where(
            Report.job_id == job_id,
            MediaItem.user_id == current_user.id
        )
    )
    report = result.scalar_one_or_none()
    
    if not report:
        # Check if job exists and is complete, then create report
        result = await db.execute(
            select(AnalysisJob)
            .options(selectinload(AnalysisJob.media_item))
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
        
        if job.status != "DONE":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Analysis must be completed before generating report"
            )

        # Create new report
        report = Report(
            job_id=job_id,
            generated_at=datetime.utcnow(),
            summary=_generate_placeholder_summary(job),
            full_report=_generate_placeholder_full_report(job)
        )
        db.add(report)
        await db.commit()
        await db.refresh(report)
    
    # Generate PDF if not exists
    if not report.pdf_path or not Path(report.pdf_path).exists():
        job = report.analysis_job
        
        # Get segment data from job results
        segments = []
        if job and job.results:
            segments = job.results.get("segments", [])
        
        # Generate PDF
        pdf_path = generate_pdf_report(
            job_id=str(job_id),
            overall_score=job.overall_score or 0.0,
            label=job.label or "UNKNOWN",
            results=job.results or {},
            summary_text=report.summary,
            media_type=job.media_item.media_type if job and job.media_item else "video"
        )
        
        # Save PDF path
        report.pdf_path = pdf_path
        await db.commit()
    
    return FileResponse(
        path=report.pdf_path,
        media_type="application/pdf",
        filename=f"deepfakeshield_report_{job_id}.pdf"
    )



@router.get("/{job_id}/report.json")
async def get_report_json(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Download the report as JSON."""
    result = await db.execute(
        select(Report)
        .join(AnalysisJob)
        .join(MediaItem)
        .where(
            Report.job_id == job_id,
            MediaItem.user_id == current_user.id
        )
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    return JSONResponse(
        content=report.full_report or {},
        headers={
            "Content-Disposition": f"attachment; filename=deepfakeshield_report_{job_id}.json"
        }
    )


@router.get("/", response_model=list[ReportResponse])
async def list_reports(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """List all reports for the current user."""
    result = await db.execute(
        select(Report)
        .options(selectinload(Report.analysis_job).selectinload(AnalysisJob.media_item))
        .join(AnalysisJob)
        .join(MediaItem)
        .where(MediaItem.user_id == current_user.id)
        .order_by(Report.generated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    reports = result.scalars().all()
    
    response = []
    for r in reports:
        response.append(ReportResponse(
            id=r.id,
            job_id=r.job_id,
            summary=r.summary,
            full_report=r.full_report,
            generated_at=r.generated_at,
            created_at=r.created_at,
            overall_score=r.analysis_job.overall_score if r.analysis_job else 0.0,
            media_type=r.analysis_job.media_item.media_type if r.analysis_job and r.analysis_job.media_item else "unknown"
        ))
        
    return response


def _generate_placeholder_summary(job: AnalysisJob) -> str:
    """Generate a placeholder summary (replace with LLM call)."""
    score = job.overall_score or 0
    label = job.label or "UNKNOWN"
    
    if label == "AUTHENTIC":
        return f"This analysis suggests the media is likely authentic with a confidence score of {(1-score)*100:.1f}%. No significant manipulation indicators were detected across video, audio, and lip-sync analysis."
    elif label == "LIKELY_FAKE":
        return f"This analysis indicates the media may be manipulated with a suspicion score of {score*100:.1f}%. Several indicators suggest potential deepfake manipulation. We recommend further verification."
    else:
        return f"This analysis detected strong manipulation indicators with a suspicion score of {score*100:.1f}%. The media is likely a deepfake. Please verify the source and consider additional forensic analysis."


def _generate_placeholder_full_report(job: AnalysisJob) -> dict:
    """Generate a placeholder full report."""
    return {
        "version": "1.0.0",
        "job_id": str(job.id),
        "generated_at": datetime.utcnow().isoformat(),
        "verdict": {
            "label": job.label,
            "overall_score": job.overall_score,
            "confidence": "high" if job.overall_score and job.overall_score > 0.8 else "medium"
        },
        "analysis": {
            "video": job.results.get("video", {}) if job.results else {},
            "audio": job.results.get("audio", {}) if job.results else {},
            "lipsync": job.results.get("lipsync", {}) if job.results else {},
        },
        "recommendations": [
            "Verify the original source of this media",
            "Check for additional context or metadata",
            "Consider consulting with forensic experts for high-stakes decisions"
        ],
        "limitations": [
            "AI detection is not 100% accurate",
            "Results should be considered alongside other evidence",
            "Detection performance may vary with compression and quality"
        ]
    }
