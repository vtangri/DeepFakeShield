"""
Multimodal Fusion Service.
Combines video, audio, and lip-sync signals for final verdict.
"""
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import BaseInferenceService


class MultimodalFusionService(BaseInferenceService):
    """
    Multimodal fusion for combining detection signals.
    Uses learnable weights or cross-attention for calibrated predictions.
    """
    
    MODEL_VERSION = "v1.0.0"
    
    # Default modality weights
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
        self.weights = weights or self.DEFAULT_WEIGHTS
    
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
        
        Args:
            input_data: Dict with 'video', 'audio', 'lipsync' results
        """
        video_result = input_data.get("video", {})
        audio_result = input_data.get("audio", {})
        lipsync_result = input_data.get("lipsync", {})
        
        return {
            "video_score": video_result.get("score", 0.0),
            "audio_score": audio_result.get("score", 0.0),
            "lipsync_score": lipsync_result.get("score", 0.0),
            "video_confidence": video_result.get("confidence", 1.0),
            "audio_confidence": audio_result.get("confidence", 1.0),
            "lipsync_confidence": lipsync_result.get("confidence", 1.0),
            "raw_results": input_data,
        }
    
    def predict(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute fused score."""
        video_score = preprocessed_data["video_score"]
        audio_score = preprocessed_data["audio_score"]
        lipsync_score = preprocessed_data["lipsync_score"]
        
        video_conf = preprocessed_data["video_confidence"]
        audio_conf = preprocessed_data["audio_confidence"]
        lipsync_conf = preprocessed_data["lipsync_confidence"]
        
        if self.model is not None and TORCH_AVAILABLE:
            # Learned fusion
            with torch.no_grad():
                features = torch.tensor([
                    video_score, audio_score, lipsync_score,
                    video_conf, audio_conf, lipsync_conf
                ]).float().unsqueeze(0).to(self.device)
                
                output = self.model(features)
                fused_score = float(output.squeeze().cpu().numpy())
        else:
            # Confidence-weighted fusion
            adjusted_weights = {
                "video": self.weights["video"] * video_conf,
                "audio": self.weights["audio"] * audio_conf,
                "lipsync": self.weights["lipsync"] * lipsync_conf,
            }
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            
            fused_score = (
                adjusted_weights["video"] * video_score +
                adjusted_weights["audio"] * audio_score +
                adjusted_weights["lipsync"] * lipsync_score
            )
        
        # Check for agreement/disagreement between modalities
        scores = [video_score, audio_score, lipsync_score]
        agreement = 1 - np.std(scores)
        
        return {
            "fused_score": float(fused_score),
            "agreement": float(agreement),
            "modality_scores": {
                "video": video_score,
                "audio": audio_score,
                "lipsync": lipsync_score,
            },
            "used_weights": self.weights,
        }
    
    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final verdict."""
        score = raw_output["fused_score"]
        agreement = raw_output["agreement"]
        modality_scores = raw_output["modality_scores"]
        
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
        
        # Calculate confidence based on agreement
        confidence = agreement * (1 - abs(score - 0.5) * 0.5)
        
        # Identify primary concerns
        concerns = []
        if modality_scores["video"] > 0.5:
            concerns.append("Visual manipulation artifacts detected")
        if modality_scores["audio"] > 0.5:
            concerns.append("Synthetic audio patterns detected")
        if modality_scores["lipsync"] > 0.5:
            concerns.append("Audio-visual synchronization mismatch")
        
        return {
            "overall_score": float(score),
            "label": label,
            "description": description,
            "confidence": float(confidence),
            "agreement": float(agreement),
            "modality_breakdown": modality_scores,
            "concerns": concerns,
            "weights_used": raw_output["used_weights"],
        }
    
    def calibrate(self, validation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calibrate fusion weights using validation data.
        
        Args:
            validation_data: List of dicts with modality scores and ground truth
        
        Returns:
            Optimized weights
        """
        # Simple grid search for optimal weights
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
                        weights["video"] * sample["video_score"] +
                        weights["audio"] * sample["audio_score"] +
                        weights["lipsync"] * sample["lipsync_score"]
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
            "fusion_type": "learned" if self.model else "weighted",
            "weights": self.weights,
        })
        return info
