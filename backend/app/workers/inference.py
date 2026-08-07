"""
Inference Celery worker tasks for ML model execution.

These workers run real ML inference using the services in ml/inference/.
They properly handle edge cases like missing audio tracks and still images
by returning None scores for inapplicable modalities.
"""
from pathlib import Path
from typing import Dict, Any, List
import json
import sys
import time
import logging

from celery import shared_task
import numpy as np

from app.core.celery_app import celery_app, TaskState
from app.core.config import settings
from app.db import SessionLocal, update_job_status
from app.models import AnalysisJob, ModelRun, Segment

# Add repo root to path to resolve ml imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.inference.video_forensics import VideoForensicsService
from ml.inference.audio_spoof import AudioSpoofService
from ml.inference.lipsync import LipSyncService
from ml.inference.fusion import MultimodalFusionService

logger = logging.getLogger(__name__)


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
            return {"job_id": job_id, "video_score": None, "note": "No frames to analyze"}

        start_time = time.time()

        # Instantiate and run the VideoForensicsService
        video_service = VideoForensicsService(
            model_path=settings.VIDEO_MODEL_PATH,
            device="cuda" if settings.ENABLE_GPU else "cpu",
        )
        video_service.load_model()
        preprocessed = video_service.preprocess(frames_data)
        raw_predictions = video_service.predict(preprocessed)
        video_result = video_service.postprocess(raw_predictions)

        score = video_result["score"]
        predictions = video_result.get("predictions", [])
        inference_mode = video_result.get("inference_mode", "unknown")

        inference_time_ms = int((time.time() - start_time) * 1000)

        # Record model run
        add_model_run(
            job_id=job_id,
            model_name=f"video_forensics_vit_{inference_mode}",
            model_version=settings.VIDEO_MODEL_VERSION,
            score=float(score),
            predictions={"frame_predictions": predictions[:50], "mode": inference_mode},
            inference_time_ms=inference_time_ms
        )

        # Add flagged segments for suspicious frames
        flagged_frames = [p for p in predictions if p.get("fake_probability", 0.0) > 0.7]
        for p in flagged_frames:
            add_segment(
                job_id=job_id,
                start_ms=p["timestamp_ms"],
                end_ms=p["timestamp_ms"] + 200,
                segment_type="video",
                score=p["fake_probability"],
                reason=f"Potential video manipulation detected in frame (probability: {p['fake_probability']:.2f})"
            )

        update_job_status(job_id, TaskState.INFER_VIDEO, 1.0)

        return {
            "job_id": job_id,
            "video_score": float(score),
            "confidence": video_result.get("confidence", 0.0),
            "max_score": float(max([p.get("fake_probability", 0.0) for p in predictions]) if predictions else 0.0),
            "frame_count": len(frames),
            "flagged_count": len(flagged_frames),
            "inference_mode": inference_mode,
        }

    except Exception as e:
        logger.error(f"Video inference failed: {e}", exc_info=True)
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="inference", max_retries=3)
def run_audio_inference(self, job_id: str, audio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run audio spoof detection on extracted audio.

    Skips analysis and returns audio_score=None if no audio track is present.
    """
    try:
        update_job_status(job_id, TaskState.INFER_AUDIO, 0.0)

        has_audio = audio_data.get("has_audio", False)
        audio_path = audio_data.get("audio_path")

        if not has_audio or not audio_path or not Path(audio_path).exists():
            # No audio track — skip analysis entirely
            update_job_status(job_id, TaskState.INFER_AUDIO, 1.0)
            return {
                "job_id": job_id,
                "audio_score": None,
                "note": audio_data.get("note", "No audio track present — audio analysis skipped"),
            }

        start_time = time.time()

        # Instantiate and run the AudioSpoofService
        audio_service = AudioSpoofService(
            model_path=settings.AUDIO_MODEL_PATH,
            device="cuda" if settings.ENABLE_GPU else "cpu",
        )
        audio_service.load_model()
        preprocessed = audio_service.preprocess({"audio_path": audio_path})
        raw_predictions = audio_service.predict(preprocessed)
        audio_result = audio_service.postprocess(raw_predictions)

        score = audio_result["score"]
        inference_mode = audio_result.get("inference_mode", "unknown")

        inference_time_ms = int((time.time() - start_time) * 1000)

        # Record model run
        add_model_run(
            job_id=job_id,
            model_name=f"audio_spoof_{inference_mode}",
            model_version=settings.AUDIO_MODEL_VERSION,
            score=float(score),
            predictions={
                "spoof_probability": score,
                "mode": inference_mode,
                "spectral_features": audio_result.get("spectral_features", {}),
            },
            inference_time_ms=inference_time_ms
        )

        # Add flagged segment if suspicious
        if score > 0.6:
            add_segment(
                job_id=job_id,
                start_ms=0,
                end_ms=int(audio_result.get("duration_sec", 5) * 1000),
                segment_type="audio",
                score=score,
                reason=f"Audio spectral anomaly detected (spoof probability: {score:.2f})"
            )

        update_job_status(job_id, TaskState.INFER_AUDIO, 1.0)

        return {
            "job_id": job_id,
            "audio_score": float(score),
            "confidence": audio_result.get("confidence", 0.0),
            "inference_mode": inference_mode,
        }

    except Exception as e:
        logger.error(f"Audio inference failed: {e}", exc_info=True)
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="inference", max_retries=3)
def run_lipsync_inference(self, job_id: str, frames_data: Dict,
                          audio_data: Dict, transcript: Dict) -> Dict[str, Any]:
    """Run lip-sync verification.

    Skips analysis and returns lipsync_score=None if:
    - No audio track is present
    - No faces are detected in the video
    - Media is a still image
    """
    try:
        update_job_status(job_id, TaskState.LIPSYNC, 0.0)

        has_audio = audio_data.get("has_audio", False)
        audio_path = audio_data.get("audio_path")

        if not has_audio:
            update_job_status(job_id, TaskState.LIPSYNC, 1.0)
            return {
                "job_id": job_id,
                "lipsync_score": None,
                "note": "No audio track — lip-sync analysis requires both audio and video",
            }

        frames = frames_data.get("frames", [])
        if len(frames) < 3:
            update_job_status(job_id, TaskState.LIPSYNC, 1.0)
            return {
                "job_id": job_id,
                "lipsync_score": None,
                "note": f"Insufficient frames ({len(frames)}) — lip-sync analysis requires video with multiple frames",
            }

        start_time = time.time()

        # Instantiate and run the LipSyncService
        lipsync_service = LipSyncService()
        lipsync_service.load_model()
        preprocessed = lipsync_service.preprocess({
            "frames": frames_data,
            "audio_path": audio_path,
            "transcript": transcript,
        })
        raw_predictions = lipsync_service.predict(preprocessed)
        lipsync_result = lipsync_service.postprocess(raw_predictions)

        score = lipsync_result.get("score")  # Can be None if analysis was skipped
        analyzed = lipsync_result.get("analyzed", False)

        inference_time_ms = int((time.time() - start_time) * 1000)

        if analyzed and score is not None:
            add_model_run(
                job_id=job_id,
                model_name="lipsync_cross_correlation",
                model_version="v1.0.0",
                score=float(score),
                predictions={
                    "mismatch_score": score,
                    "sync_offset_ms": lipsync_result.get("sync_offset_ms", 0),
                    "correlation": lipsync_result.get("correlation", 0),
                },
                inference_time_ms=inference_time_ms
            )

            if score > 0.5:
                add_segment(
                    job_id=job_id,
                    start_ms=0,
                    end_ms=5000,
                    segment_type="lipsync",
                    score=score,
                    reason=f"Lip-audio synchronization mismatch detected (offset: {lipsync_result.get('sync_offset_ms', 0):.0f}ms)"
                )

        update_job_status(job_id, TaskState.LIPSYNC, 1.0)

        return {
            "job_id": job_id,
            "lipsync_score": float(score) if score is not None else None,
            "confidence": lipsync_result.get("confidence", 0.0) if analyzed else 0.0,
            "note": lipsync_result.get("note", "") if not analyzed else "",
        }

    except Exception as e:
        logger.error(f"Lip-sync inference failed: {e}", exc_info=True)
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="inference", max_retries=3)
def run_fusion(self, job_id: str, video_result: Dict, audio_result: Dict,
               lipsync_result: Dict) -> Dict[str, Any]:
    """Run multimodal fusion to get final score.

    Handles None scores from skipped modalities by dynamically
    recalibrating weights among active modalities only.
    """
    try:
        update_job_status(job_id, TaskState.FUSION, 0.0)

        video_score = video_result.get("video_score")
        audio_score = audio_result.get("audio_score")
        lipsync_score = lipsync_result.get("lipsync_score")

        # Instantiate and run MultimodalFusionService
        fusion_service = MultimodalFusionService()
        fusion_service.load_model()

        preprocessed = fusion_service.preprocess({
            "video": {
                "score": video_score,
                "confidence": video_result.get("confidence", 1.0) if video_score is not None else 0.0,
            },
            "audio": {
                "score": audio_score,
                "confidence": audio_result.get("confidence", 1.0) if audio_score is not None else 0.0,
            },
            "lipsync": {
                "score": lipsync_score,
                "confidence": lipsync_result.get("confidence", 1.0) if lipsync_score is not None else 0.0,
            },
        })
        raw_predictions = fusion_service.predict(preprocessed)
        fusion_result = fusion_service.postprocess(raw_predictions)

        overall_score = fusion_result["overall_score"]
        label = fusion_result["label"]
        weights = fusion_result.get("weights_used", {})
        active_modalities = fusion_result.get("active_modalities", [])

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
                        "description": fusion_result.get("description", ""),
                        "weights": weights,
                        "active_modalities": active_modalities,
                        "concerns": fusion_result.get("concerns", []),
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
            "active_modalities": active_modalities,
        }

    except Exception as e:
        logger.error(f"Fusion failed: {e}", exc_info=True)
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="inference")
def run_inference_pipeline(self, preprocess_results: Dict, job_id: str):
    """Run the full inference pipeline.

    Properly passes audio metadata downstream so that audio/lipsync
    analysis is skipped when there is no audio track.
    """
    try:
        frames_data = preprocess_results.get("frames", {})
        audio_data = preprocess_results.get("audio", {})
        transcript = preprocess_results.get("transcript", {})

        # Call task bodies directly via .run() rather than .apply(...).get() — calling
        # .get() from within a running Celery task is explicitly disallowed (deadlock risk)
        # and these sub-steps only ever run inline as part of this pipeline anyway.
        video_result = run_video_inference.run(job_id, frames_data)
        audio_result = run_audio_inference.run(job_id, audio_data)
        lipsync_result = run_lipsync_inference.run(job_id, frames_data, audio_data, transcript)
        fusion_result = run_fusion.run(job_id, video_result, audio_result, lipsync_result)

        return fusion_result

    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise
