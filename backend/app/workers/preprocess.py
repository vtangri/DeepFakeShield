"""
Preprocessing Celery worker tasks.
"""
import os
import subprocess
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from celery import shared_task
import cv2
import numpy as np

from app.core.celery_app import celery_app, TaskState
from app.core.config import settings
from app.db import SessionLocal, update_job_status
from app.models import AnalysisJob, MediaItem


@celery_app.task(bind=True, queue="preprocess", max_retries=3)
def validate_media(self, job_id: str) -> Dict[str, Any]:
    """Validate the uploaded media file."""
    try:
        update_job_status(job_id, TaskState.VALIDATING, 0.0)
        
        db = SessionLocal()
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            media = db.query(MediaItem).filter(MediaItem.id == job.media_id).first()
            if not media:
                raise ValueError(f"Media item not found for job {job_id}")
            
            file_path = Path(media.storage_path)
            if not file_path.exists():
                raise ValueError(f"Media file not found: {file_path}")
            
            # Verify file hash
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            
            if sha256_hash.hexdigest() != media.sha256:
                raise ValueError("File hash mismatch - file may be corrupted")
            
            update_job_status(job_id, TaskState.VALIDATING, 1.0)
            
            return {
                "job_id": job_id,
                "media_id": str(media.id),
                "file_path": str(file_path),
                "media_type": media.media_type,
            }
        finally:
            db.close()
            
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="preprocess", max_retries=3)
def extract_frames(self, job_id: str, file_path: str, fps: int = 5) -> Dict[str, Any]:
    """Extract frames from video at specified FPS."""
    try:
        update_job_status(job_id, TaskState.EXTRACTING, 0.0)
        
        file_path = Path(file_path)
        output_dir = file_path.parent / f"frames_{job_id}"
        output_dir.mkdir(exist_ok=True)
        
        # Use OpenCV to extract frames
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {file_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((total_frames / video_fps) * 1000) if video_fps > 0 else 0
        
        # Calculate frame interval
        frame_interval = max(1, int(video_fps / fps))
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = output_dir / f"frame_{extracted_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append({
                    "path": str(frame_path),
                    "timestamp_ms": int((frame_count / video_fps) * 1000),
                    "frame_number": frame_count
                })
                extracted_count += 1
            
            frame_count += 1
            progress = frame_count / total_frames if total_frames > 0 else 0
            if frame_count % 100 == 0:
                update_job_status(job_id, TaskState.EXTRACTING, progress * 0.5)
        
        cap.release()
        update_job_status(job_id, TaskState.EXTRACTING, 0.5)
        
        return {
            "job_id": job_id,
            "frames_dir": str(output_dir),
            "frame_count": extracted_count,
            "duration_ms": duration_ms,
            "frames": frames,
        }
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="preprocess", max_retries=3)
def extract_audio(self, job_id: str, file_path: str) -> Dict[str, Any]:
    """Extract audio track from video/audio file.
    
    Returns has_audio=False if the media has no audio track, instead of
    failing. This allows downstream tasks to skip audio analysis.
    """
    try:
        update_job_status(job_id, TaskState.EXTRACTING, 0.5)
        
        file_path = Path(file_path)
        audio_path = file_path.parent / f"audio_{job_id}.wav"
        
        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg", "-y",
            "-i", str(file_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16kHz
            "-ac", "1",  # Mono
            str(audio_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if audio was actually extracted
        # ffmpeg may succeed but produce an empty/tiny file if no audio track exists
        has_audio = (
            result.returncode == 0
            and audio_path.exists()
            and audio_path.stat().st_size > 1000  # > 1KB means real audio data
        )
        
        if not has_audio:
            # No audio track in this media — this is NOT an error
            update_job_status(job_id, TaskState.EXTRACTING, 1.0)
            return {
                "job_id": job_id,
                "audio_path": None,
                "has_audio": False,
                "note": "No audio track detected in media file",
            }
        
        update_job_status(job_id, TaskState.EXTRACTING, 1.0)
        
        return {
            "job_id": job_id,
            "audio_path": str(audio_path),
            "has_audio": True,
        }
        
    except Exception as e:
        # ffmpeg failure is not fatal — just means no audio analysis
        update_job_status(job_id, TaskState.EXTRACTING, 1.0)
        return {
            "job_id": job_id,
            "audio_path": None,
            "has_audio": False,
            "note": f"Audio extraction failed: {str(e)}",
        }


@celery_app.task(bind=True, queue="preprocess", max_retries=3)
def transcribe_audio(self, job_id: str, audio_path: str) -> Dict[str, Any]:
    """Transcribe audio using Whisper."""
    try:
        update_job_status(job_id, TaskState.TRANSCRIBING, 0.0)
        
        # Import whisper (optional dependency)
        try:
            import whisper
        except ImportError:
            # Return empty transcript if Whisper not available
            update_job_status(job_id, TaskState.TRANSCRIBING, 1.0)
            return {
                "job_id": job_id,
                "transcript": {
                    "full_text": "",
                    "words": [],
                }
            }
        
        # Load model
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en"
        )
        
        # Extract word-level timestamps
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info["word"].strip(),
                    "start_ms": int(word_info["start"] * 1000),
                    "end_ms": int(word_info["end"] * 1000),
                    "confidence": word_info.get("probability", 0.0)
                })
        
        update_job_status(job_id, TaskState.TRANSCRIBING, 1.0)
        
        return {
            "job_id": job_id,
            "transcript": {
                "full_text": result.get("text", ""),
                "words": words,
            }
        }
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise


@celery_app.task(bind=True, queue="preprocess")
def run_preprocessing_pipeline(self, job_id: str):
    """Run the full preprocessing pipeline.
    
    Handles three media types:
    - video: extract frames + attempt audio extraction + transcription
    - audio: transcription only
    - image: single-frame analysis only (no audio, no lip-sync)
    """
    try:
        # Validate
        validation = validate_media.apply(args=[job_id]).get(disable_sync_subtasks=False)
        
        file_path = validation["file_path"]
        media_type = validation["media_type"]
        
        results = {"job_id": job_id, "media_type": media_type}
        
        if media_type == "video":
            # Extract frames
            frames_result = extract_frames.apply(args=[job_id, file_path]).get(disable_sync_subtasks=False)
            results["frames"] = frames_result
            
            # Extract audio (may return has_audio=False if no audio track)
            audio_result = extract_audio.apply(args=[job_id, file_path]).get(disable_sync_subtasks=False)
            results["audio"] = audio_result
            
            # Only transcribe if audio was actually found
            if audio_result.get("has_audio") and audio_result.get("audio_path"):
                transcript_result = transcribe_audio.apply(
                    args=[job_id, audio_result["audio_path"]]
                ).get(disable_sync_subtasks=False)
                results["transcript"] = transcript_result["transcript"]
            else:
                results["transcript"] = {"full_text": "", "words": []}
            
        elif media_type == "audio":
            # Audio-only: transcribe directly
            results["audio"] = {"audio_path": file_path, "has_audio": True}
            transcript_result = transcribe_audio.apply(args=[job_id, file_path]).get(disable_sync_subtasks=False)
            results["transcript"] = transcript_result["transcript"]
            results["frames"] = {"frames": [], "frame_count": 0}
            
        elif media_type == "image":
            # Image: single frame, no audio, no lip-sync
            results["frames"] = {
                "frames": [{"path": file_path, "timestamp_ms": 0, "frame_number": 0}],
                "frame_count": 1,
            }
            results["audio"] = {"audio_path": None, "has_audio": False, "note": "Image media — no audio"}
            results["transcript"] = {"full_text": "", "words": []}
            
        else:
            # Unknown media type — try video processing
            frames_result = extract_frames.apply(args=[job_id, file_path]).get(disable_sync_subtasks=False)
            results["frames"] = frames_result
            results["audio"] = {"audio_path": None, "has_audio": False}
            results["transcript"] = {"full_text": "", "words": []}
        
        # Update job with preprocessing results
        db = SessionLocal()
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.results = results
                db.commit()
        finally:
            db.close()
        
        return results
        
    except Exception as e:
        update_job_status(job_id, TaskState.FAILED, 0.0, str(e))
        raise
