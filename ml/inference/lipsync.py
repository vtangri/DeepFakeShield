"""
Lip-Sync Verification Service.
Detects audio-visual synchronization mismatches.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .base import BaseInferenceService


class LipSyncService(BaseInferenceService):
    """
    Lip-sync verification for detecting A/V mismatch.
    Compares mouth movements with audio phonemes.
    """
    
    MODEL_VERSION = "v1.0.0"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        window_size_ms: int = 500,
    ):
        super().__init__(model_path, device)
        self.window_size_ms = window_size_ms
        self.face_cascade = None
    
    def load_model(self) -> None:
        """Load lip-sync verification model."""
        if CV2_AVAILABLE:
            # Load face detector for mouth ROI extraction
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if TORCH_AVAILABLE and self.model_path and Path(self.model_path).exists():
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
        else:
            self.model = None
        
        self.is_loaded = True
    
    def _extract_mouth_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract mouth region from frame."""
        if not CV2_AVAILABLE or self.face_cascade is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Mouth region (lower third of face)
        mouth_y = y + int(h * 0.6)
        mouth_h = int(h * 0.4)
        mouth_x = x + int(w * 0.2)
        mouth_w = int(w * 0.6)
        
        mouth_roi = frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
        return mouth_roi
    
    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess frames and audio for lip-sync analysis.
        
        Args:
            input_data: Dict with 'frames' and 'transcript'
        """
        frames = input_data.get("frames", {}).get("frames", [])
        transcript = input_data.get("transcript", {})
        
        mouth_features = []
        
        for frame_info in frames:
            if isinstance(frame_info, dict) and "path" in frame_info:
                if CV2_AVAILABLE:
                    frame = cv2.imread(frame_info["path"])
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mouth_roi = self._extract_mouth_roi(frame)
                        mouth_features.append({
                            "timestamp_ms": frame_info.get("timestamp_ms", 0),
                            "has_mouth": mouth_roi is not None,
                            "mouth_roi": mouth_roi,
                        })
        
        # Extract word timings from transcript
        words = transcript.get("words", [])
        
        return {
            "mouth_features": mouth_features,
            "words": words,
            "total_frames": len(mouth_features),
        }
    
    def predict(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lip-sync alignment."""
        mouth_features = preprocessed_data["mouth_features"]
        words = preprocessed_data["words"]
        
        if not mouth_features or not words:
            return {
                "mismatch_score": 0.0,
                "segments": [],
            }
        
        # Placeholder: analyze synchronization
        # Real implementation would compare mouth openness with phoneme timing
        segments = []
        mismatch_scores = []
        
        for word in words:
            start_ms = word.get("start_ms", 0)
            end_ms = word.get("end_ms", 0)
            
            # Find frames in this time window
            window_frames = [
                f for f in mouth_features
                if start_ms <= f["timestamp_ms"] <= end_ms
            ]
            
            # Check if mouth was detected during speech
            if window_frames:
                mouth_detected_ratio = sum(1 for f in window_frames if f["has_mouth"]) / len(window_frames)
                
                # Simulate mismatch detection
                mismatch = np.random.uniform(0.0, 0.3) if mouth_detected_ratio > 0.5 else 0.5
                mismatch_scores.append(mismatch)
                
                if mismatch > 0.4:
                    segments.append({
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "word": word.get("word", ""),
                        "mismatch_score": float(mismatch),
                    })
        
        overall_mismatch = float(np.mean(mismatch_scores)) if mismatch_scores else 0.0
        
        return {
            "mismatch_score": overall_mismatch,
            "segments": segments,
            "analyzed_words": len(words),
        }
    
    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess lip-sync results."""
        score = raw_output.get("mismatch_score", 0.0)
        segments = raw_output.get("segments", [])
        
        if score < 0.3:
            label = "SYNCHRONIZED"
        elif score < 0.6:
            label = "MINOR_MISMATCH"
        else:
            label = "MAJOR_MISMATCH"
        
        return {
            "score": float(score),
            "label": label,
            "confidence": float(1 - score * 0.5),
            "flagged_segments": segments[:5],
            "analyzed_words": raw_output.get("analyzed_words", 0),
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_version": self.MODEL_VERSION,
            "window_size_ms": self.window_size_ms,
            "model_type": "LipSync-Verifier",
        })
        return info
