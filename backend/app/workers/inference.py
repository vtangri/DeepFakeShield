"""
Inference Celery worker tasks for ML model execution.
"""
from pathlib import Path
from typing import Dict, Any, List
import json

from celery import shared_task
import numpy as np

from app.core.celery_app import celery_app, TaskState
from app.core.config import settings
from app.db.session import SessionLocal
from app.models import AnalysisJob, ModelRun, Segment


def update_job_status(job_id: str, stage: str, progress: float, error: str = None):
    """Update job status in database."""
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.stage = stage
            job.status = stage
            job.progress = progress
            if error:
                job.error_message = error
                job.status = TaskState.FAILED
            db.commit()
    finally:
        db.close()


def add_model_run(job_id: str, model_name: str, model_version: str, 
                  score: float, predictions: dict, inference_time_ms: int):
    """Record a model run in the database."""
    db = SessionLocal()
    try:
        model_run = ModelRun(
            job_id=job_id,
            model_name=model_name,
            model_version=model_version,
            score=score,
            predictions=predictions,
            inference_time_ms=inference_time_ms
        )
        db.add(model_run)
        db.commit()
    finally:
        db.close()


def add_segment(job_id: str, start_ms: int, end_ms: int, 
                segment_type: str, score: float, reason: str):
    """Add a flagged segment to the database."""
    db = SessionLocal()
    try:
        segment = Segment(
            job_id=job_id,
            start_ms=start_ms,
            end_ms=end_ms,
            segment_type=segment_type,
            score=score,
            reason=reason
        )
        db.add(segment)
        db.commit()
    finally:
        db.close()


@celery_app.task(bind=True, queue="inference", max_retries=3)
def run_video_inference(self, job_id: str, frames_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run video forensics inference on extracted frames."""
    try:
        update_job_status(job_id, TaskState.INFER_VIDEO, 0.0)
        
        frames = frames_data.get("frames", [])
        if not frames:
            update_job_status(job_id, TaskState.INFER_VIDEO, 1.0)
            return {"job_id": job_id, "video_score": 0.0, "predictions": []}
        
        # TODO: Load actual ViT/Swin model
        # For now, simulate inference
        import time
        import random
        
        start_time = time.time()
        
        predictions = []
        flagged_frames = []
        
        for i, frame_info in enumerate(frames):
            # Simulate frame-level prediction
            fake_prob = random.uniform(0.0, 0.5)  # Simulate mostly authentic
            
            predictions.append({
                "frame_number": frame_info["frame_number"],
                "timestamp_ms": frame_info["timestamp_ms"],
                "fake_probability": fake_prob,
            })
            
            if fake_prob > 0.7:
                flagged_frames.append(frame_info)
            
            if i % 10 == 0:
                progress = i / len(frames)
                update_job_status(job_id, TaskState.INFER_VIDEO, progress)
        
        # Calculate aggregate score
        avg_score = np.mean([p["fake_probability"] for p in predictions])
        max_score = np.max([p["fake_probability"] for p in predictions])
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        # Record model run
        add_model_run(
            job_id=job_id,
            model_name="video_forensics_vit",
            model_version=settings.VIDEO_MODEL_VERSION,
            score=float(avg_score),
            predictions={"frame_predictions": predictions[:100]},  # Limit stored predictions
            inference_time_ms=inference_time_ms
        )
        
        # Add flagged segments
        for frame in flagged_frames:
            add_segment(
                job_id=job_id,
                start_ms=frame["timestamp_ms"],
                end_ms=frame["timestamp_ms"] + 200,  # ~200ms per frame at 5fps
                segment_type="video",
                score=0.75,
                reason="Potential manipulation detected in frame"
            )
        
        update_job_status(job_id, TaskState.INFER_VIDEO, 1.0)
        
        return {
            "job_id": job_id,
            "video_score": float(avg_score),
            "max_score": float(max_score),
            "frame_count": len(frames),
            "flagged_count": len(flagged_frames),
        }
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="inference", max_retries=3)
def run_audio_inference(self, job_id: str, audio_path: str) -> Dict[str, Any]:
    """Run audio spoof detection on extracted audio."""
    try:
        update_job_status(job_id, TaskState.INFER_AUDIO, 0.0)
        
        if not audio_path or not Path(audio_path).exists():
            update_job_status(job_id, TaskState.INFER_AUDIO, 1.0)
            return {"job_id": job_id, "audio_score": 0.0}
        
        # TODO: Load actual AASIST model
        # For now, simulate inference
        import time
        import random
        
        start_time = time.time()
        
        # Simulate audio analysis
        time.sleep(0.5)  # Simulate processing time
        
        spoof_probability = random.uniform(0.0, 0.4)  # Simulate mostly authentic
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        # Record model run
        add_model_run(
            job_id=job_id,
            model_name="audio_spoof_aasist",
            model_version=settings.AUDIO_MODEL_VERSION,
            score=float(spoof_probability),
            predictions={"spoof_probability": spoof_probability},
            inference_time_ms=inference_time_ms
        )
        
        # Add flagged segment if suspicious
        if spoof_probability > 0.6:
            add_segment(
                job_id=job_id,
                start_ms=0,
                end_ms=5000,  # First 5 seconds
                segment_type="audio",
                score=spoof_probability,
                reason="Audio spectral anomaly detected"
            )
        
        update_job_status(job_id, TaskState.INFER_AUDIO, 1.0)
        
        return {
            "job_id": job_id,
            "audio_score": float(spoof_probability),
        }
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="inference", max_retries=3)
def run_lipsync_inference(self, job_id: str, frames_data: Dict, 
                          transcript: Dict) -> Dict[str, Any]:
    """Run lip-sync verification."""
    try:
        update_job_status(job_id, TaskState.LIPSYNC, 0.0)
        
        # TODO: Implement actual lip-sync verification
        # For now, simulate
        import time
        import random
        
        start_time = time.time()
        time.sleep(0.3)
        
        mismatch_score = random.uniform(0.0, 0.3)
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        add_model_run(
            job_id=job_id,
            model_name="lipsync_verifier",
            model_version="v1.0.0",
            score=float(mismatch_score),
            predictions={"mismatch_score": mismatch_score},
            inference_time_ms=inference_time_ms
        )
        
        if mismatch_score > 0.5:
            add_segment(
                job_id=job_id,
                start_ms=2000,
                end_ms=4000,
                segment_type="lipsync",
                score=mismatch_score,
                reason="Lip-audio synchronization mismatch"
            )
        
        update_job_status(job_id, TaskState.LIPSYNC, 1.0)
        
        return {
            "job_id": job_id,
            "lipsync_score": float(mismatch_score),
        }
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="inference", max_retries=3)
def run_fusion(self, job_id: str, video_result: Dict, audio_result: Dict,
               lipsync_result: Dict) -> Dict[str, Any]:
    """Run multimodal fusion to get final score."""
    try:
        update_job_status(job_id, TaskState.FUSION, 0.0)
        
        video_score = video_result.get("video_score", 0.0)
        audio_score = audio_result.get("audio_score", 0.0)
        lipsync_score = lipsync_result.get("lipsync_score", 0.0)
        
        # Simple weighted fusion (replace with actual fusion model)
        weights = {"video": 0.5, "audio": 0.3, "lipsync": 0.2}
        
        overall_score = (
            weights["video"] * video_score +
            weights["audio"] * audio_score +
            weights["lipsync"] * lipsync_score
        )
        
        # Determine label
        if overall_score < 0.3:
            label = "AUTHENTIC"
        elif overall_score < 0.6:
            label = "LIKELY_FAKE"
        else:
            label = "FAKE"
        
        # Update job with final results
        from datetime import datetime
        db = SessionLocal()
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.overall_score = overall_score
                job.label = label
                job.results = {
                    **(job.results or {}),
                    "video": video_result,
                    "audio": audio_result,
                    "lipsync": lipsync_result,
                    "fusion": {
                        "overall_score": overall_score,
                        "label": label,
                        "weights": weights,
                    }
                }
                db.commit()
        finally:
            db.close()
        
        update_job_status(job_id, TaskState.FUSION, 1.0)
        
        return {
            "job_id": job_id,
            "overall_score": overall_score,
            "label": label,
            "video_score": video_score,
            "audio_score": audio_score,
            "lipsync_score": lipsync_score,
        }
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="inference")
def run_inference_pipeline(self, job_id: str, preprocess_results: Dict):
    """Run the full inference pipeline."""
    try:
        frames_data = preprocess_results.get("frames", {})
        audio_data = preprocess_results.get("audio", {})
        transcript = preprocess_results.get("transcript", {})
        
        # Run video inference
        video_result = run_video_inference.apply(
            args=[job_id, frames_data]
        ).get()
        
        # Run audio inference
        audio_path = audio_data.get("audio_path", "")
        audio_result = run_audio_inference.apply(
            args=[job_id, audio_path]
        ).get()
        
        # Run lip-sync inference
        lipsync_result = run_lipsync_inference.apply(
            args=[job_id, frames_data, transcript]
        ).get()
        
        # Run fusion
        fusion_result = run_fusion.apply(
            args=[job_id, video_result, audio_result, lipsync_result]
        ).get()
        
        return fusion_result
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise
