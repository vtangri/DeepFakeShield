"""
Report generation Celery worker tasks.
"""
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

from celery import shared_task

from app.core.celery_app import celery_app, TaskState
from app.core.config import settings
from app.db import SessionLocal, update_job_status
from app.models import AnalysisJob, Report


# LLM System Prompt for Report Generation
REPORT_SYSTEM_PROMPT = """You are DeepFakeShield Report Assistant.
You write forensic-style summaries of an AI deepfake analysis.

You must:
- Use cautious language (likely, suggests, indicates)
- Highlight evidence and timestamps
- Include limitations and next verification steps

You must NOT:
- Provide instructions to create deepfakes
- Explain how to bypass detection
- Give advice on evading moderation

Output format:
1. Summary verdict with confidence
2. Evidence highlights with timestamps
3. Modality breakdown
4. Limitations
5. Recommended verification steps
"""


def generate_report_with_llm(analysis_results: Dict[str, Any]) -> str:
    """Generate report using LLM (placeholder implementation)."""
    
    # Check if OpenAI API key is configured
    if not settings.OPENAI_API_KEY:
        return _generate_fallback_report(analysis_results)
    
    try:
        import openai
        
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a forensic report for this analysis:\n{json.dumps(analysis_results, indent=2)}"}
            ],
            max_tokens=1000,
            temperature=0.3,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return _generate_fallback_report(analysis_results)


def _generate_fallback_report(results: Dict[str, Any]) -> str:
    """Generate a fallback report without LLM."""
    overall_score = results.get("overall_score", 0)
    label = results.get("label", "UNKNOWN")
    video_score = results.get("video_score")
    audio_score = results.get("audio_score")
    lipsync_score = results.get("lipsync_score")
    active_modalities = results.get("active_modalities", [])
    
    # Determine verdict text
    if label == "AUTHENTIC":
        verdict = "This media appears to be authentic."
        confidence = "high"
    elif label == "LIKELY_FAKE":
        verdict = "This media shows some indicators of potential manipulation."
        confidence = "medium"
    elif label == "FAKE":
        verdict = "This media shows strong indicators of manipulation."
        confidence = "high"
    else:
        verdict = "This media shows some anomalies that warrant further review."
        confidence = "medium"
    
    # Format scores, showing N/A for skipped modalities
    video_line = f"- **Score:** {video_score:.1%}" if video_score is not None else "- **Score:** N/A (not applicable for this media type)"
    audio_line = f"- **Score:** {audio_score:.1%}" if audio_score is not None else "- **Score:** N/A (no audio track detected)"
    lipsync_line = f"- **Score:** {lipsync_score:.1%}" if lipsync_score is not None else "- **Score:** N/A (requires both audio and visible faces)"
    
    video_detail = ("No significant anomalies detected" if video_score is not None and video_score < 0.5 
                    else "Potential manipulation indicators found" if video_score is not None
                    else "Video analysis was not performed")
    audio_detail = ("Audio appears authentic" if audio_score is not None and audio_score < 0.5
                    else "Audio shows potential synthesis patterns" if audio_score is not None
                    else "No audio track was present in the media")
    lipsync_detail = ("Lip movements align with audio" if lipsync_score is not None and lipsync_score < 0.5
                     else "Potential lip-sync mismatch detected" if lipsync_score is not None
                     else "Lip-sync analysis requires both audio and video with visible faces")
    
    report = f"""# DeepFakeShield Analysis Report

## Summary Verdict

{verdict}

**Overall Suspicion Score:** {overall_score:.1%}
**Classification:** {label}
**Confidence:** {confidence}
**Modalities Analyzed:** {', '.join(active_modalities) if active_modalities else 'None'}

## Modality Analysis

### Video Analysis
{video_line}
- Analyzed video frames for manipulation artifacts
- {video_detail}

### Audio Analysis  
{audio_line}
- Analyzed audio for synthetic speech markers
- {audio_detail}

### Lip-Sync Analysis
{lipsync_line}
- Verified audio-visual synchronization
- {lipsync_detail}

## Limitations

⚠️ **Important considerations:**

1. AI detection is not 100% accurate
2. Results should be considered alongside other evidence
3. Detection performance may vary with compression and quality
4. Novel deepfake techniques may not be detected
5. Modalities marked N/A were not analyzed due to media properties

## Recommended Next Steps

1. **Verify the source**: Check the original publication source
2. **Request original file**: Compressed versions may affect analysis
3. **Cross-reference**: Compare with known authentic media from the same source
4. **Consult experts**: For high-stakes decisions, consider forensic expert review

---
*Generated by DeepFakeShield AI on {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}*
"""
    
    return report


@celery_app.task(bind=True, queue="default", max_retries=3)
def generate_report(self, fusion_results: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Generate the final forensic report."""
    try:
        update_job_status(job_id, TaskState.REPORT, 0.0)
        
        # Generate report text
        report_text = generate_report_with_llm(fusion_results)
        
        update_job_status(job_id, TaskState.REPORT, 0.5)
        
        # Create full report JSON
        full_report = {
            "version": "1.0.0",
            "job_id": job_id,
            "generated_at": datetime.utcnow().isoformat(),
            "verdict": {
                "label": fusion_results.get("label"),
                "overall_score": fusion_results.get("overall_score"),
            },
            "analysis": {
                "video_score": fusion_results.get("video_score"),
                "audio_score": fusion_results.get("audio_score"),
                "lipsync_score": fusion_results.get("lipsync_score"),
            },
            "report_text": report_text,
        }
        
        # Save report to database
        db = SessionLocal()
        try:
            report = db.query(Report).filter(Report.job_id == job_id).first()
            
            if not report:
                report = Report(job_id=job_id)
                db.add(report)
            
            report.summary = report_text
            report.full_report = full_report
            report.generated_at = datetime.utcnow()
            report.llm_model_used = settings.LLM_MODEL if settings.OPENAI_API_KEY else "fallback"
            
            db.commit()
        finally:
            db.close()
        
        update_job_status(job_id, TaskState.REPORT, 1.0)
        
        return {
            "job_id": job_id,
            "report_generated": True,
        }
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="default")
def finalize_job(self, report_result: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Finalize the analysis job."""
    try:
        from datetime import datetime
        
        db = SessionLocal()
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = TaskState.DONE
                job.stage = TaskState.DONE
                job.progress = 1.0
                job.completed_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
        
        return {
            "job_id": job_id,
            "status": TaskState.DONE,
        }
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="default")
def run_full_pipeline(self, job_id: str):
    """Run the complete analysis pipeline using a Celery canvas chain (non-blocking)."""
    from app.workers.preprocess import run_preprocessing_pipeline
    from app.workers.inference import run_inference_pipeline
    from celery import chain
    
    try:
        # Build non-blocking pipeline chain. Task args are swapped on downstream tasks so chain piped inputs match:
        # 1. run_preprocessing_pipeline(job_id) -> returns preprocess_results
        # 2. run_inference_pipeline(preprocess_results, job_id) -> returns fusion_results
        # 3. generate_report(fusion_results, job_id) -> returns report_result
        # 4. finalize_job(report_result, job_id) -> returns final_result
        pipeline = chain(
            run_preprocessing_pipeline.s(job_id),
            run_inference_pipeline.s(job_id),
            generate_report.s(job_id),
            finalize_job.s(job_id)
        )
        
        # Dispatch the chained pipeline asynchronously
        pipeline.apply_async()
        return {"job_id": job_id, "status": "QUEUED"}
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise
