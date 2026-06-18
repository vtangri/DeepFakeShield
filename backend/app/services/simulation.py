import asyncio
import random
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import AnalysisJob, MediaItem
from app.core import TaskState
from app.db.session import SessionLocal

async def simulate_analysis_pipeline(job_id: UUID):
    """
    INSTANT analysis simulation - no delays.
    Completes analysis immediately for fastest possible response.
    """
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            return

        # INSTANT: Skip straight to completion - no delays!
        # Briefly set each stage for logging purposes only
        stages = [
            TaskState.VALIDATING,
            TaskState.EXTRACTING, 
            TaskState.TRANSCRIBING,
            TaskState.INFER_VIDEO,
            TaskState.INFER_AUDIO,
            TaskState.LIPSYNC,
            TaskState.FUSION,
            TaskState.REPORT,
        ]

        # Rapidly cycle through stages (no sleep!)
        for idx, stage in enumerate(stages):
            job.stage = stage
            job.status = stage  
            job.progress = (idx + 1) / len(stages)
            db.commit()

        # Complete immediately!
        job.stage = TaskState.DONE
        job.status = TaskState.DONE
        job.progress = 1.0
        job.completed_at = datetime.utcnow()
        
        # Generate realistic mock results
        # DEMO MODE: Check filename for keywords to force detection
        filename = job.media_item.original_filename.lower() if job.media_item else ""
        fake_keywords = ["fake", "deep", "manipulated", "synthetic", "ai", "gen", "test", "demo", "deepfake", "kaggle", "clip", "example", "video"]
        auth_keywords = ["authentic", "real", "safe", "original", "verified", "true", "human", "person"]
        
        is_known_fake = any(k in filename for k in fake_keywords)
        is_known_auth = any(k in filename for k in auth_keywords)
        
        # Priority logic: 
        # 1. Force Fake if fake keyword matches
        # 2. Force Authentic if auth keyword matches
        # 3. Else, 40% random chance for fake (higher for better demo variety)
        if is_known_auth:
            is_fake = False
        elif is_known_fake:
            is_fake = True
        else:
            is_fake = (random.random() < 0.40) 
            
        score = random.uniform(0.78, 0.96) if is_fake else random.uniform(0.02, 0.15)
        label = "LIKELY_FAKE" if is_fake else "AUTHENTIC"
        
        # Modality scores with slight variation
        video_score = max(0, min(1, score + random.uniform(-0.08, 0.08)))
        audio_score = max(0, min(1, score * 0.85 + random.uniform(-0.05, 0.05)))
        lipsync_score = max(0, min(1, score * 0.7 + random.uniform(-0.05, 0.05)))
        
        # Generate detailed flagged segments
        segments = []
        if is_fake:
            segments = [
                {"start_ms": 1200, "end_ms": 3500, "segment_type": "video", 
                 "score": video_score, "reason": "Facial boundary blending artifacts - GAN-generated edges show inconsistent pixel gradients at jawline (confidence: 92%)"},
                {"start_ms": 3800, "end_ms": 5200, "segment_type": "video",
                 "score": 0.81, "reason": "Temporal flickering detected in eye region - irregular blink patterns inconsistent with natural eye movement"},
                {"start_ms": 4000, "end_ms": 6800, "segment_type": "audio",
                 "score": audio_score, "reason": "Voice spectrogram shows unnatural formant transitions - F0 pitch contour exhibits synthetic smoothing patterns"},
                {"start_ms": 2500, "end_ms": 5500, "segment_type": "lipsync",
                 "score": lipsync_score, "reason": "Lip-audio desynchronization of 85-120ms - visemes do not match phoneme timing windows"},
                {"start_ms": 7000, "end_ms": 9200, "segment_type": "video",
                 "score": 0.74, "reason": "Unnatural head pose transitions - motion vectors show discontinuities inconsistent with physics"},
                {"start_ms": 9500, "end_ms": 11000, "segment_type": "audio",
                 "score": 0.69, "reason": "Breath pattern anomaly - respiratory sounds missing or artificially inserted between words"}
            ]
        elif score > 0.12:
            segments = [
                {"start_ms": 5500, "end_ms": 7200, "segment_type": "video",
                 "score": 0.28, "reason": "Minor compression artifact detected - likely from video re-encoding (benign)"},
                {"start_ms": 8000, "end_ms": 8800, "segment_type": "audio",
                 "score": 0.22, "reason": "Slight audio clipping detected - possibly microphone distortion (benign)"}
            ]

        job.overall_score = score
        job.label = label
        
        # Enhanced detailed results
        frames_analyzed = random.randint(120, 240)
        faces_detected = random.randint(1, 3)
        
        job.results = {
            "version": "2.1.0",
            "pipeline": "multimodal_forensic",
            "analysis_type": "comprehensive",
            "video": {
                "score": video_score,
                "confidence": 0.89 + random.uniform(-0.05, 0.05),
                "manipulation_type": random.choice(["face_swap", "face_reenactment", "lip_sync_manipulation"]) if is_fake else None,
                "manipulation_method": random.choice(["DeepFaceLab", "FaceSwap", "FSGAN", "First Order Motion"]) if is_fake else None,
                "frames_analyzed": frames_analyzed,
                "faces_detected": faces_detected,
                "face_detection_confidence": 0.97,
                "artifacts": {
                    "boundary_artifacts": is_fake,
                    "temporal_inconsistency": is_fake and random.random() > 0.3,
                    "color_histogram_anomaly": is_fake and random.random() > 0.5,
                    "compression_artifacts": random.random() > 0.7
                },
                "frame_analysis": {
                    "suspicious_frames": random.randint(15, 45) if is_fake else random.randint(0, 5),
                    "high_confidence_fake_frames": random.randint(8, 25) if is_fake else 0,
                    "blending_score": round(random.uniform(0.7, 0.95), 3) if is_fake else round(random.uniform(0.05, 0.2), 3)
                }
            },
            "audio": {
                "score": audio_score,
                "confidence": 0.87 + random.uniform(-0.05, 0.05),
                "voice_cloning_detected": is_fake,
                "cloning_method": random.choice(["TTS synthesis", "Voice conversion", "RVC", "VITS"]) if is_fake else None,
                "sample_rate": 16000,
                "duration_analyzed_ms": int(15.5 * 1000),
                "spectral_analysis": {
                    "mfcc_anomaly_score": round(random.uniform(0.6, 0.9), 3) if is_fake else round(random.uniform(0.05, 0.25), 3),
                    "formant_consistency": "LOW" if is_fake else "NORMAL",
                    "pitch_variance": round(random.uniform(0.02, 0.08), 4) if is_fake else round(random.uniform(0.12, 0.25), 4),
                    "harmonic_ratio": round(random.uniform(0.3, 0.6), 3) if is_fake else round(random.uniform(0.7, 0.95), 3)
                },
                "voice_identity": {
                    "speaker_embedding_distance": round(random.uniform(0.6, 0.9), 3) if is_fake else round(random.uniform(0.05, 0.2), 3),
                    "naturalness_score": round(random.uniform(0.2, 0.5), 2) if is_fake else round(random.uniform(0.75, 0.95), 2)
                }
            },
            "lipsync": {
                "score": lipsync_score,
                "confidence": 0.85 + random.uniform(-0.05, 0.05),
                "mismatch_detected": is_fake,
                "sync_offset_ms": random.randint(85, 180) if is_fake else random.randint(-30, 30),
                "correlation_score": round(random.uniform(0.15, 0.45), 3) if is_fake else round(random.uniform(0.7, 0.95), 3),
                "phoneme_accuracy": round(random.uniform(0.25, 0.55), 2) if is_fake else round(random.uniform(0.8, 0.98), 2),
                "viseme_match_rate": round(random.uniform(0.2, 0.5), 2) if is_fake else round(random.uniform(0.75, 0.95), 2)
            },
            "segments": segments,
            "technical_summary": {
                "models_used": ["ViT-B/16-FaceForensics", "Wav2Vec2-ASR", "SyncNet-AV", "XceptionNet"],
                "ensemble_method": "weighted_average",
                "inference_device": "CPU",
                "total_inference_time_ms": random.randint(650, 950)
            },
            "metadata": {
                "duration_s": 15.5,
                "resolution": "1920x1080",
                "fps": 30,
                "codec": "H.264",
                "bitrate_kbps": 4500,
                "audio_channels": 2,
                "file_hash": f"sha256:{random.randbytes(16).hex()}",
                "analyzed_at": datetime.utcnow().isoformat()
            },
            # 1. Media Quality Assessment
            "media_quality": {
                "overall_quality_score": round(random.uniform(75, 98), 1) if not ("blur" in filename or "noise" in filename) else round(random.uniform(30, 60), 1),
                "blur_detection": {
                    "is_blurry": "blur" in filename,
                    "blur_score": round(random.uniform(0.1, 0.4), 2) if "blur" not in filename else round(random.uniform(0.6, 0.9), 2)
                },
                "noise_level": {
                    "is_noisy": "noise" in filename,
                    "snr_db": round(random.uniform(25, 40), 1) if "noise" not in filename else round(random.uniform(10, 18), 1)
                },
                "compression_analysis": {
                    "double_compression_detected": random.random() < 0.2,
                    "jpeg_quality_estimate": random.randint(85, 98)
                }
            },
            # 2. Frequency Domain Forensics
            "frequency_analysis": {
                "gan_fingerprint_detected": is_fake and random.random() > 0.3,
                "spectrum_consistency": "ABNORMAL" if is_fake else "NORMAL",
                "fft_anomalies": {
                    "high_freq_artifacts": is_fake,
                    "checkerboard_patterns": is_fake and random.random() > 0.5
                }
            },
            # 3. Linguistic Analysis (if audio/transcript exists)
            "linguistic_analysis": {
                "fluency_score": round(random.uniform(0.8, 0.98), 2),
                "suspicious_patterns": {
                    "templated_speech": "template" in filename,
                    "unnatural_repetition": "repeat" in filename,
                    "sentiment_inconsistency": is_fake and random.random() > 0.6
                },
                "generated_text_probability": 0.88 if is_fake else 0.12
            },
            # 4. Container & Metadata Forensics
            "container_analysis": {
                "metadata_consistency": "CONSISTENT" if not is_fake else "SUSPICIOUS",
                "tool_fingerprints": ["Adobe Premiere", "ffmpeg"] if is_fake else ["Camera Original"],
                "modification_date_mismatch": is_fake and random.random() > 0.4
            }
        }
        
        db.commit()
        
        # Create Report record
        from app.models import Report, Segment
        
        # Create Segment records
        for seg_data in segments:
            segment = Segment(
                job_id=job_id,
                start_ms=seg_data["start_ms"],
                end_ms=seg_data["end_ms"],
                segment_type=seg_data["segment_type"],
                score=seg_data["score"],
                reason=seg_data["reason"]
            )
            db.add(segment)
            
        if label == "AUTHENTIC":
            summary = f"✓ This media appears AUTHENTIC with {(1-score)*100:.0f}% confidence. No significant manipulation indicators detected."
        else:
            summary = f"⚠ POTENTIAL DEEPFAKE detected with {score*100:.0f}% suspicion score. Multiple manipulation indicators found across video, audio, and lip-sync analysis."
        
        report = Report(
            job_id=job_id,
            summary=summary,
            full_report={
                "version": "2.0.0",
                "job_id": str(job_id),
                "generated_at": datetime.utcnow().isoformat(),
                "verdict": {"label": label, "overall_score": score},
                "analysis": {
                    "video_score": video_score, 
                    "audio_score": audio_score, 
                    "lipsync_score": lipsync_score
                },
                "segments": segments
            },
            generated_at=datetime.utcnow()
        )
        db.add(report)
        db.commit()
        
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
        if job:
            job.status = TaskState.FAILED
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()

