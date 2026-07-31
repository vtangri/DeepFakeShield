"""
Multimodal Fusion Service.
Combines video, audio, and lip-sync signals for final verdict.

Dynamically recalibrates weights when modalities are unavailable
(e.g., no audio track, no visible faces for lip-sync).
"""
import logging
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import BaseInferenceService

logger = logging.getLogger(__name__)


class MultimodalFusionService(BaseInferenceService):
    """
    Multimodal fusion for combining detection signals.

    Uses confidence-weighted averaging with dynamic recalibration.
    When a modality is unavailable (score is None), its weight is
    redistributed proportionally among the remaining active modalities.
    """

    MODEL_VERSION = "v1.0.0"

    # Default modality weights (tuned via grid search on validation set)
    DEFAULT_WEIGHTS = {
        "video": 0.45,
        "audio": 0.30,
        "lipsync": 0.25,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(model_path, device)
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def load_model(self) -> None:
        """Load fusion model (if using learned fusion)."""
        if TORCH_AVAILABLE and self.model_path:
            from pathlib import Path
            if Path(self.model_path).exists():
                self.model = torch.load(self.model_path, map_location=self.device)
                self.model.eval()

        self.is_loaded = True

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare modality scores for fusion.

        Handles None scores for unavailable modalities.

        Args:
            input_data: Dict with 'video', 'audio', 'lipsync' results.
                        Each value is a dict with 'score' (float or None)
                        and 'confidence'.
        """
        video_result = input_data.get("video", {})
        audio_result = input_data.get("audio", {})
        lipsync_result = input_data.get("lipsync", {})

        # Extract scores — None means modality was not applicable
        video_score = video_result.get("score")
        audio_score = audio_result.get("score")
        lipsync_score = lipsync_result.get("score")

        # Extract confidence values
        video_conf = video_result.get("confidence", 1.0) if video_score is not None else 0.0
        audio_conf = audio_result.get("confidence", 1.0) if audio_score is not None else 0.0
        lipsync_conf = lipsync_result.get("confidence", 1.0) if lipsync_score is not None else 0.0

        return {
            "video_score": video_score,
            "audio_score": audio_score,
            "lipsync_score": lipsync_score,
            "video_confidence": video_conf,
            "audio_confidence": audio_conf,
            "lipsync_confidence": lipsync_conf,
            "raw_results": input_data,
        }

    def predict(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute fused score with dynamic weight recalibration."""
        video_score = preprocessed_data["video_score"]
        audio_score = preprocessed_data["audio_score"]
        lipsync_score = preprocessed_data["lipsync_score"]

        video_conf = preprocessed_data["video_confidence"]
        audio_conf = preprocessed_data["audio_confidence"]
        lipsync_conf = preprocessed_data["lipsync_confidence"]

        # Build active modalities dict (skip None scores)
        active_scores = {}
        active_weights = {}
        active_confidences = {}

        if video_score is not None:
            active_scores["video"] = video_score
            active_weights["video"] = self.weights["video"]
            active_confidences["video"] = video_conf

        if audio_score is not None:
            active_scores["audio"] = audio_score
            active_weights["audio"] = self.weights["audio"]
            active_confidences["audio"] = audio_conf

        if lipsync_score is not None:
            active_scores["lipsync"] = lipsync_score
            active_weights["lipsync"] = self.weights["lipsync"]
            active_confidences["lipsync"] = lipsync_conf

        if not active_scores:
            return {
                "fused_score": 0.0,
                "agreement": 0.0,
                "modality_scores": {"video": None, "audio": None, "lipsync": None},
                "used_weights": {},
                "active_modalities": [],
                "note": "No modalities available for fusion",
            }

        # Confidence-weighted fusion with normalization
        adjusted_weights = {
            k: active_weights[k] * active_confidences[k]
            for k in active_scores
        }

        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        else:
            # Equal weights fallback
            n = len(active_scores)
            normalized_weights = {k: 1.0 / n for k in active_scores}

        fused_score = sum(
            normalized_weights[k] * active_scores[k]
            for k in active_scores
        )

        # Agreement metric: how much do active modalities agree?
        scores_list = list(active_scores.values())
        if len(scores_list) > 1:
            agreement = float(1 - np.std(scores_list))
        else:
            agreement = 1.0  # Single modality → full "agreement"

        return {
            "fused_score": float(fused_score),
            "agreement": float(agreement),
            "modality_scores": {
                "video": video_score,
                "audio": audio_score,
                "lipsync": lipsync_score,
            },
            "used_weights": normalized_weights,
            "active_modalities": list(active_scores.keys()),
        }

    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final verdict."""
        score = raw_output["fused_score"]
        agreement = raw_output["agreement"]
        modality_scores = raw_output["modality_scores"]
        active = raw_output.get("active_modalities", [])

        # Determine label
        if score < 0.25:
            label = "AUTHENTIC"
            description = "No significant manipulation indicators detected."
        elif score < 0.45:
            label = "LIKELY_AUTHENTIC"
            description = "Minor anomalies detected but likely authentic."
        elif score < 0.6:
            label = "SUSPICIOUS"
            description = "Some manipulation indicators present. Further review recommended."
        elif score < 0.8:
            label = "LIKELY_FAKE"
            description = "Strong manipulation indicators detected."
        else:
            label = "FAKE"
            description = "High confidence of manipulation across modalities."

        # Note which modalities were analyzed
        if len(active) < 3:
            skipped = [m for m in ["video", "audio", "lipsync"] if m not in active]
            description += f" Note: {', '.join(skipped)} analysis was not applicable for this media."

        # Calculate confidence based on agreement and number of modalities
        modality_factor = len(active) / 3.0  # More modalities = higher confidence
        confidence = agreement * modality_factor * (1 - abs(score - 0.5) * 0.5)

        # Identify primary concerns
        concerns = []
        if modality_scores.get("video") is not None and modality_scores["video"] > 0.5:
            concerns.append("Visual manipulation artifacts detected")
        if modality_scores.get("audio") is not None and modality_scores["audio"] > 0.5:
            concerns.append("Synthetic audio patterns detected")
        if modality_scores.get("lipsync") is not None and modality_scores["lipsync"] > 0.5:
            concerns.append("Audio-visual synchronization mismatch")

        return {
            "overall_score": float(score),
            "label": label,
            "description": description,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "agreement": float(agreement),
            "modality_breakdown": modality_scores,
            "concerns": concerns,
            "weights_used": raw_output["used_weights"],
            "active_modalities": active,
        }

    def calibrate(self, validation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calibrate fusion weights using validation data.

        Args:
            validation_data: List of dicts with modality scores and ground truth
        """
        best_weights = self.DEFAULT_WEIGHTS.copy()
        best_accuracy = 0.0

        for v_w in np.arange(0.2, 0.7, 0.1):
            for a_w in np.arange(0.1, 0.5, 0.1):
                l_w = 1.0 - v_w - a_w
                if l_w < 0.05:
                    continue

                weights = {"video": v_w, "audio": a_w, "lipsync": l_w}
                correct = 0

                for sample in validation_data:
                    fused = (
                        weights["video"] * sample.get("video_score", 0) +
                        weights["audio"] * sample.get("audio_score", 0) +
                        weights["lipsync"] * sample.get("lipsync_score", 0)
                    )
                    predicted = 1 if fused > 0.5 else 0
                    if predicted == sample["label"]:
                        correct += 1

                accuracy = correct / len(validation_data) if validation_data else 0
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights

        self.weights = best_weights
        return best_weights

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_version": self.MODEL_VERSION,
            "fusion_type": "learned" if self.model else "confidence_weighted",
            "weights": self.weights,
        })
        return info
