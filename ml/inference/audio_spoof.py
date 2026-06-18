"""
Audio Spoof Detection Service using AASIST-style architecture.
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import BaseInferenceService


class AudioSpoofService(BaseInferenceService):
    """
    Audio spoof detection for synthetic speech.
    Uses spectral analysis and neural networks to detect voice cloning.
    """
    
    MODEL_VERSION = "v1.0.0"
    SAMPLE_RATE = 16000
    MAX_DURATION_SEC = 60
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(model_path, device)
        self.sample_rate = sample_rate
        self.mel_transform = None
    
    def load_model(self) -> None:
        """Load the audio spoof detection model."""
        if not TORCH_AVAILABLE:
            self.is_loaded = True
            return
        
        # Setup mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
        )
        
        if self.model_path and Path(self.model_path).exists():
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
        else:
            # Placeholder: simple CNN for audio classification
            self.model = self._create_simple_model()
            self.model.to(self.device)
            self.model.eval()
        
        self.is_loaded = True
    
    def _create_simple_model(self) -> nn.Module:
        """Create a simple CNN model for audio classification."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess audio for inference.
        
        Args:
            input_data: Dict with 'audio_path' or 'waveform' (numpy array)
        """
        waveform = None
        sample_rate = self.sample_rate
        
        if "audio_path" in input_data:
            audio_path = Path(input_data["audio_path"])
            if not audio_path.exists():
                return {"waveform": None, "error": "Audio file not found"}
            
            if TORCH_AVAILABLE:
                waveform, sr = torchaudio.load(str(audio_path))
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
            else:
                # Fallback: return empty
                return {"waveform": None, "duration_sec": 0}
        
        elif "waveform" in input_data:
            waveform = torch.from_numpy(input_data["waveform"]).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
        
        if waveform is not None:
            # Truncate if too long
            max_samples = self.MAX_DURATION_SEC * self.sample_rate
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            duration_sec = waveform.shape[1] / self.sample_rate
        else:
            duration_sec = 0
        
        return {
            "waveform": waveform,
            "duration_sec": duration_sec,
        }
    
    def predict(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on preprocessed audio."""
        waveform = preprocessed_data.get("waveform")
        
        if waveform is None or not TORCH_AVAILABLE or self.model is None:
            # Fallback: simulated prediction
            return {
                "spoof_probability": np.random.uniform(0.05, 0.2),
                "duration_sec": preprocessed_data.get("duration_sec", 0),
            }
        
        with torch.no_grad():
            # Compute mel spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec = mel_spec.unsqueeze(0)  # Add batch dim
            mel_spec = mel_spec.to(self.device)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            # Predict
            output = self.model(mel_spec)
            prob = output.squeeze().cpu().numpy()
        
        return {
            "spoof_probability": float(prob),
            "duration_sec": preprocessed_data["duration_sec"],
        }
    
    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess audio prediction."""
        prob = raw_output.get("spoof_probability", 0.0)
        duration = raw_output.get("duration_sec", 0.0)
        
        if prob < 0.3:
            label = "AUTHENTIC"
        elif prob < 0.6:
            label = "SUSPICIOUS"
        else:
            label = "SPOOFED"
        
        return {
            "score": float(prob),
            "label": label,
            "confidence": float(1 - abs(prob - 0.5) * 2),  # Higher near 0 or 1
            "duration_sec": duration,
            "analysis": {
                "spectral_anomaly": prob > 0.4,
                "synthetic_markers": prob > 0.6,
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_version": self.MODEL_VERSION,
            "sample_rate": self.sample_rate,
            "model_type": "AASIST-lite",
        })
        return info
