"""
Lip-Sync Verification Service.

Detects audio-visual synchronization mismatches by computing the
cross-correlation between mouth openness (extracted from video frames)
and audio energy envelope.  This is a signal-processing approach
that requires NO trained neural network weights.

References:
- Chung & Zisserman (2016), "Out of Time: Automated Lip Sync in the Wild"
- Haliassos et al. (2021), "Lips Don't Lie"
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .base import BaseInferenceService

logger = logging.getLogger(__name__)


class LipSyncService(BaseInferenceService):
    """
    Lip-sync verification for detecting audio-visual mismatch.

    Algorithm:
    1. Extract mouth ROI from each video frame using face detection
    2. Compute mouth openness signal (vertical extent of mouth region)
    3. Compute audio energy envelope at matching timestamps
    4. Cross-correlate the two signals to find synchronization offset
    5. Mismatches > 80ms indicate dubbing or deepfake reenactment
    """

    MODEL_VERSION = "v1.0.0"
    SYNC_THRESHOLD_MS = 80  # Offsets above this are flagged

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
        """Load face detector for mouth ROI extraction."""
        if CV2_AVAILABLE:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Lip-sync service loaded (signal-processing mode)")
        else:
            logger.warning("OpenCV not available — lip-sync analysis will be limited")

        self.model = None  # No neural network needed
        self.is_loaded = True

    def _extract_mouth_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract mouth region from a video frame."""
        if not CV2_AVAILABLE or self.face_cascade is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        if len(faces) == 0:
            return None

        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Mouth region: lower 40% of face bounding box
        mouth_y = y + int(h * 0.6)
        mouth_h = int(h * 0.4)
        mouth_x = x + int(w * 0.2)
        mouth_w = int(w * 0.6)

        mouth_roi = frame[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]
        return mouth_roi

    def _compute_mouth_openness(self, mouth_roi: np.ndarray) -> float:
        """
        Estimate mouth openness from a mouth ROI image.

        Converts to grayscale, thresholds to find dark regions (inside of
        mouth), and measures the vertical extent of the dark area relative
        to the ROI height.
        """
        if mouth_roi is None or mouth_roi.size == 0:
            return 0.0

        if len(mouth_roi.shape) == 3:
            gray = cv2.cvtColor(mouth_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = mouth_roi

        # Adaptive threshold to find dark interior of mouth
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find vertical extent of dark region (mouth opening)
        col_sum = np.sum(binary > 0, axis=1)
        open_rows = np.where(col_sum > binary.shape[1] * 0.3)[0]

        if len(open_rows) == 0:
            return 0.0

        vertical_extent = (open_rows[-1] - open_rows[0]) / binary.shape[0]
        return float(np.clip(vertical_extent, 0.0, 1.0))

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract mouth openness signal and audio energy from input data.

        Args:
            input_data: Dict with:
                - 'frames': frame data (list of dicts with 'path', 'timestamp_ms')
                - 'audio_path': path to extracted audio file (optional)
                - 'transcript': word-level transcript (optional)
        """
        # Extract frames data
        frames_data = input_data.get("frames", {})
        if isinstance(frames_data, dict):
            frames_list = frames_data.get("frames", [])
        elif isinstance(frames_data, list):
            frames_list = frames_data
        else:
            frames_list = []

        audio_path = input_data.get("audio_path")
        transcript = input_data.get("transcript", {})

        # Extract mouth openness signal from frames
        mouth_openness = []
        timestamps_ms = []
        face_detected_count = 0

        for frame_info in frames_list:
            ts = frame_info.get("timestamp_ms", 0)
            timestamps_ms.append(ts)

            if isinstance(frame_info, dict) and "path" in frame_info and CV2_AVAILABLE:
                frame = cv2.imread(frame_info["path"])
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mouth_roi = self._extract_mouth_roi(frame)
                    if mouth_roi is not None:
                        openness = self._compute_mouth_openness(mouth_roi)
                        mouth_openness.append(openness)
                        face_detected_count += 1
                    else:
                        mouth_openness.append(0.0)
                else:
                    mouth_openness.append(0.0)
            else:
                mouth_openness.append(0.0)

        # Compute audio energy envelope at matching timestamps
        audio_energy = []
        has_audio = False

        if audio_path and Path(audio_path).exists():
            try:
                if LIBROSA_AVAILABLE:
                    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
                    if len(y) > 0:
                        has_audio = True
                        # Compute RMS energy for each frame timestamp
                        for ts in timestamps_ms:
                            sample_idx = int(ts * sr / 1000)
                            window = 1600  # 100ms window
                            start = max(0, sample_idx - window // 2)
                            end = min(len(y), sample_idx + window // 2)
                            if end > start:
                                rms = float(np.sqrt(np.mean(y[start:end] ** 2)))
                            else:
                                rms = 0.0
                            audio_energy.append(rms)
                else:
                    # No librosa — try basic file size check
                    if Path(audio_path).stat().st_size > 1000:
                        has_audio = True
            except Exception as e:
                logger.warning(f"Failed to load audio for lip-sync: {e}")

        return {
            "mouth_openness": mouth_openness,
            "mouth_features": mouth_openness,  # Expose for unit test compatibility
            "audio_energy": audio_energy,
            "timestamps_ms": timestamps_ms,
            "has_audio": has_audio,
            "face_detected_count": face_detected_count,
            "total_frames": len(frames_list),
            "words": transcript.get("words", []),
        }

    def predict(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lip-sync alignment using cross-correlation."""
        mouth_openness = preprocessed_data["mouth_openness"]
        audio_energy = preprocessed_data["audio_energy"]
        has_audio = preprocessed_data["has_audio"]
        face_count = preprocessed_data["face_detected_count"]
        total_frames = preprocessed_data["total_frames"]

        # Guard: cannot do lip-sync without both audio and visible faces
        if not has_audio:
            return {
                "mismatch_score": None,
                "sync_offset_ms": None,
                "correlation": None,
                "note": "No audio track present — lip-sync analysis skipped",
                "analyzed": False,
            }

        if face_count < 3:
            return {
                "mismatch_score": None,
                "sync_offset_ms": None,
                "correlation": None,
                "note": f"Insufficient face detections ({face_count}/{total_frames} frames) — lip-sync analysis skipped",
                "analyzed": False,
            }

        if not audio_energy or len(audio_energy) != len(mouth_openness):
            return {
                "mismatch_score": None,
                "sync_offset_ms": None,
                "correlation": None,
                "note": "Audio/video length mismatch — lip-sync analysis skipped",
                "analyzed": False,
            }

        # Normalize signals
        mo = np.array(mouth_openness, dtype=np.float64)
        ae = np.array(audio_energy, dtype=np.float64)

        mo = (mo - mo.mean()) / (mo.std() + 1e-8)
        ae = (ae - ae.mean()) / (ae.std() + 1e-8)

        # Cross-correlate to find sync offset
        correlation = np.correlate(mo, ae, mode="full")
        mid = len(correlation) // 2

        # Find peak correlation and its lag
        best_lag = np.argmax(correlation) - mid
        peak_corr = float(correlation[np.argmax(correlation)])
        zero_lag_corr = float(correlation[mid])

        # Estimate frame interval in ms
        if len(preprocessed_data["timestamps_ms"]) >= 2:
            frame_interval_ms = (
                preprocessed_data["timestamps_ms"][-1] - preprocessed_data["timestamps_ms"][0]
            ) / max(1, len(preprocessed_data["timestamps_ms"]) - 1)
        else:
            frame_interval_ms = 200  # Default 5fps

        sync_offset_ms = abs(best_lag * frame_interval_ms)

        # Compute mismatch score
        # High zero-lag correlation = good sync, low = mismatch
        # Large sync offset = mismatch
        corr_score = max(0.0, 1.0 - max(0.0, zero_lag_corr))  # 0 = perfect sync
        offset_score = min(1.0, sync_offset_ms / 200.0)  # Normalized offset

        mismatch_score = float(np.clip(0.5 * corr_score + 0.5 * offset_score, 0.0, 1.0))

        # Find segments with poor local correlation
        segments = []
        window = max(5, len(mo) // 10)
        for i in range(0, len(mo) - window, window // 2):
            local_mo = mo[i:i + window]
            local_ae = ae[i:i + window]
            local_corr = float(np.corrcoef(local_mo, local_ae)[0, 1])
            if np.isnan(local_corr):
                local_corr = 0.0

            if local_corr < 0.2:  # Poor local correlation
                start_ms = preprocessed_data["timestamps_ms"][i] if i < len(preprocessed_data["timestamps_ms"]) else 0
                end_ms = preprocessed_data["timestamps_ms"][min(i + window, len(preprocessed_data["timestamps_ms"]) - 1)]
                segments.append({
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "local_correlation": local_corr,
                    "mismatch_score": float(1.0 - max(0.0, local_corr)),
                })

        return {
            "mismatch_score": mismatch_score,
            "sync_offset_ms": float(sync_offset_ms),
            "correlation": zero_lag_corr,
            "peak_correlation": peak_corr,
            "best_lag_frames": int(best_lag),
            "segments": segments,
            "analyzed": True,
            "face_detection_rate": face_count / max(1, total_frames),
        }

    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess lip-sync results."""
        analyzed = raw_output.get("analyzed", "mismatch_score" in raw_output)

        if not analyzed:
            return {
                "score": None,
                "label": "NOT_APPLICABLE",
                "confidence": 0.0,
                "note": raw_output.get("note", "Lip-sync analysis was not performed"),
                "flagged_segments": [],
                "analyzed": False,
            }

        score = raw_output.get("mismatch_score", 0.0)
        segments = raw_output.get("segments", [])
        sync_offset = raw_output.get("sync_offset_ms", 0.0)

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
            "sync_offset_ms": sync_offset,
            "correlation": raw_output.get("correlation", 0.0),
            "flagged_segments": segments[:5],
            "face_detection_rate": raw_output.get("face_detection_rate", 0.0),
            "analyzed": True,
        }

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_version": self.MODEL_VERSION,
            "window_size_ms": self.window_size_ms,
            "model_type": "LipSync-CrossCorrelation",
            "sync_threshold_ms": self.SYNC_THRESHOLD_MS,
        })
        return info
