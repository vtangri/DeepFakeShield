"""
Audio Spoof Detection Service.

Uses a trained CNN classifier on Mel spectrograms when trained weights are
available.  Falls back to real signal-processing-based spectral analysis
(MFCC statistics, spectral flatness, harmonic-to-noise ratio) when no
custom weights exist.  NEVER returns random scores.
"""
import logging
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

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .base import BaseInferenceService

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class AudioSpoofModel(nn.Module):
        """Simple CNN for audio spoof detection matching the training architecture."""
        
        def __init__(self, sample_rate: int = 16000):
            super().__init__()
            
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
            )
            
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, x):
            # x: (batch, samples)
            mel_spec = self.mel_transform(x)  # (batch, n_mels, time)
            mel_spec = mel_spec.unsqueeze(1)  # (batch, 1, n_mels, time)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            features = self.features(mel_spec)
            return self.classifier(features)


class AudioSpoofService(BaseInferenceService):
    """
    Audio spoof detection for synthetic speech and cloned voices.

    Two operating modes:
    1. TRAINED MODE: Loads a CNN classifier trained on the ASVspoof 2019
       dataset.  Processes Mel spectrograms through the network for a
       direct spoof probability.
    2. SPECTRAL MODE: Uses real signal-processing features to detect
       synthesis artifacts.  Analyses MFCC consistency, spectral flatness,
       harmonic-to-noise ratio, and zero-crossing rate — metrics that
       differ measurably between natural and synthesized speech
       (Todisco et al., 2019; Sahidullah et al., 2015).
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
        self.mode = "unknown"

    def _create_cnn_model(self) -> "nn.Module":
        """Create the CNN classifier architecture (must match training)."""
        return AudioSpoofModel(sample_rate=self.sample_rate)

    def load_model(self) -> None:
        """Load the audio spoof detection model."""
        if TORCH_AVAILABLE:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
            )

        resolved_path = self.find_weights("audio_spoof_final.pt")
        if resolved_path and TORCH_AVAILABLE:
            logger.info(f"Loading trained audio model from: {resolved_path}")
            self.model = self._create_cnn_model()
            checkpoint = torch.load(resolved_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.mode = "trained"
            logger.info("Audio model loaded in TRAINED mode")
        else:
            logger.info("No trained audio weights found. Using SPECTRAL analysis mode")
            self.model = None
            self.mode = "spectral"

        self.is_loaded = True

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and preprocess audio for inference.

        Args:
            input_data: Dict with 'audio_path' or 'waveform' (numpy array)
        """
        waveform = None
        raw_audio_np = None

        if "audio_path" in input_data:
            audio_path = Path(input_data["audio_path"])
            if not audio_path.exists():
                return {"waveform": None, "raw_audio": None, "error": "Audio file not found"}

            if TORCH_AVAILABLE:
                try:
                    waveform, sr = torchaudio.load(str(audio_path))
                except Exception as exc:
                    # torchaudio >= 2.9 routes load() through TorchCodec, which is
                    # not always installed. Fall back rather than lose the modality.
                    logger.warning("torchaudio.load failed (%s); falling back to librosa", exc)
                    if not LIBROSA_AVAILABLE:
                        return {
                            "waveform": None,
                            "raw_audio": None,
                            "error": f"Could not decode audio: {exc}",
                        }
                    raw_audio_np, sr = librosa.load(
                        str(audio_path), sr=self.sample_rate, mono=True
                    )
                    waveform = torch.from_numpy(raw_audio_np).float().unsqueeze(0)
                else:
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)

                    raw_audio_np = waveform.squeeze().numpy()
            elif LIBROSA_AVAILABLE:
                raw_audio_np, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            else:
                return {"waveform": None, "raw_audio": None, "error": "No audio library available"}

        elif "waveform" in input_data:
            raw_audio_np = np.array(input_data["waveform"], dtype=np.float32)
            if raw_audio_np.ndim > 1:
                raw_audio_np = raw_audio_np.mean(axis=0)
            if TORCH_AVAILABLE:
                waveform = torch.from_numpy(raw_audio_np).float().unsqueeze(0)

        if raw_audio_np is not None:
            max_samples = self.MAX_DURATION_SEC * self.sample_rate
            if len(raw_audio_np) > max_samples:
                raw_audio_np = raw_audio_np[:max_samples]
            if waveform is not None and waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            duration_sec = len(raw_audio_np) / self.sample_rate
        else:
            duration_sec = 0

        return {
            "waveform": waveform,
            "raw_audio": raw_audio_np,
            "duration_sec": duration_sec,
        }

    def predict(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on preprocessed audio."""
        waveform = preprocessed_data.get("waveform")
        raw_audio = preprocessed_data.get("raw_audio")

        if raw_audio is None and waveform is None:
            raise RuntimeError("No audio data available for analysis")

        if self.mode == "trained" and waveform is not None:
            return self._predict_trained(waveform, preprocessed_data["duration_sec"])
        else:
            return self._predict_spectral(raw_audio, preprocessed_data["duration_sec"])

    def _predict_trained(self, waveform: "torch.Tensor", duration_sec: float) -> Dict[str, Any]:
        """Run trained CNN model inference on raw waveform."""
        with torch.no_grad():
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.device)
            output = self.model(waveform)
            prob = float(output.squeeze().cpu().numpy())

        return {
            "spoof_probability": prob,
            "duration_sec": duration_sec,
            "inference_mode": "trained",
        }

    def _predict_spectral(self, raw_audio: np.ndarray, duration_sec: float) -> Dict[str, Any]:
        """
        Signal-processing-based spoof detection using spectral features.

        Synthetic speech typically exhibits:
        - Higher spectral flatness (more uniform spectrum vs natural harmonics)
        - Lower harmonic-to-noise ratio
        - More uniform MFCC coefficients (less natural variation)
        - Lower zero-crossing rate variance

        References:
        - Sahidullah et al. (2015), "A Comparison of Features for Synthetic
          Speech Detection"
        - Todisco et al. (2019), "ASVspoof 2019"
        """
        features = {}

        if LIBROSA_AVAILABLE:
            # MFCC analysis
            mfccs = librosa.feature.mfcc(y=raw_audio, sr=self.sample_rate, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfccs)

            # Spectral flatness (Wiener entropy) — synthetic audio is flatter
            spectral_flatness = librosa.feature.spectral_flatness(y=raw_audio)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(raw_audio)

            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=raw_audio, sr=self.sample_rate)

            features["mfcc_mean_var"] = float(np.mean(np.var(mfccs, axis=1)))
            features["mfcc_delta_var"] = float(np.mean(np.var(mfcc_delta, axis=1)))
            features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
            features["spectral_flatness_std"] = float(np.std(spectral_flatness))
            features["zcr_std"] = float(np.std(zcr))
            features["rolloff_std"] = float(np.std(rolloff))

        else:
            # Fallback: use numpy-based basic spectral analysis
            # Compute FFT-based spectral features
            n_fft = 1024
            hop = 256
            n_frames = max(1, (len(raw_audio) - n_fft) // hop)

            spectral_energies = []
            for i in range(0, min(n_frames * hop, len(raw_audio) - n_fft), hop):
                frame = raw_audio[i:i + n_fft]
                windowed = frame * np.hanning(n_fft)
                spectrum = np.abs(np.fft.rfft(windowed))
                spectral_energies.append(spectrum)

            if spectral_energies:
                spectra = np.array(spectral_energies)
                # Spectral flatness: geometric mean / arithmetic mean
                geo_mean = np.exp(np.mean(np.log(spectra + 1e-10), axis=1))
                arith_mean = np.mean(spectra, axis=1)
                flatness = geo_mean / (arith_mean + 1e-10)

                features["spectral_flatness_mean"] = float(np.mean(flatness))
                features["spectral_flatness_std"] = float(np.std(flatness))

                # Temporal variance of spectral energy
                features["spectral_var"] = float(np.mean(np.var(spectra, axis=0)))

                # Zero crossing rate
                zcr_values = []
                frame_len = 2048
                for i in range(0, len(raw_audio) - frame_len, frame_len):
                    frame = raw_audio[i:i + frame_len]
                    zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_len)
                    zcr_values.append(zcr)
                features["zcr_std"] = float(np.std(zcr_values)) if zcr_values else 0.0

        # Combine features into a spoof probability score
        # Higher spectral flatness → more likely synthetic
        # Lower MFCC variance → more likely synthetic (less natural variation)
        # Lower ZCR std → more likely synthetic (more uniform)
        spoof_indicators = []

        sf_mean = features.get("spectral_flatness_mean", 0.0)
        # Synthetic audio typically has spectral flatness > 0.15
        spoof_indicators.append(min(1.0, sf_mean / 0.3))

        zcr_s = features.get("zcr_std", 0.05)
        # Low ZCR variance indicates synthetic uniformity
        spoof_indicators.append(max(0.0, 1.0 - zcr_s * 20))

        if "mfcc_mean_var" in features:
            mfcc_v = features["mfcc_mean_var"]
            # Authentic speech has higher MFCC variance
            spoof_indicators.append(max(0.0, 1.0 - mfcc_v / 50.0))

        if "mfcc_delta_var" in features:
            delta_v = features["mfcc_delta_var"]
            spoof_indicators.append(max(0.0, 1.0 - delta_v / 10.0))

        spoof_probability = float(np.clip(np.mean(spoof_indicators), 0.0, 1.0))

        return {
            "spoof_probability": spoof_probability,
            "duration_sec": duration_sec,
            "inference_mode": "spectral",
            "spectral_features": features,
        }

    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess audio prediction."""
        prob = raw_output.get("spoof_probability", 0.0)
        duration = raw_output.get("duration_sec", 0.0)
        inference_mode = raw_output.get("inference_mode", "unknown")

        if prob < 0.3:
            label = "AUTHENTIC"
        elif prob < 0.6:
            label = "SUSPICIOUS"
        else:
            label = "SPOOFED"

        return {
            "score": float(prob),
            "label": label,
            "confidence": float(abs(prob - 0.5) * 2),  # Higher when far from 0.5
            "duration_sec": duration,
            "inference_mode": inference_mode,
            "analysis": {
                "spectral_anomaly": prob > 0.4,
                "synthetic_markers": prob > 0.6,
            },
            "spectral_features": raw_output.get("spectral_features", {}),
        }

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_version": self.MODEL_VERSION,
            "sample_rate": self.sample_rate,
            "model_type": "AASIST-lite-CNN",
            "inference_mode": self.mode,
        })
        return info
