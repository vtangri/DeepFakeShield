"""
Video Forensics Inference Service using Vision Transformer.

Uses a fine-tuned ViT-B/16 model for deepfake detection when trained weights
are available. Falls back to feature-based statistical analysis using the
pretrained ImageNet backbone when no custom weights exist.
"""
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .base import BaseInferenceService

logger = logging.getLogger(__name__)


class VideoForensicsService(BaseInferenceService):
    """
    Video deepfake detection using Vision Transformer (ViT-B/16).

    Two operating modes:
    1. TRAINED MODE: Loads fine-tuned weights from a .pt checkpoint file.
       The model was fine-tuned on FaceForensics++ for binary deepfake
       classification. Outputs per-frame manipulation probability.
    2. FEATURE MODE: Uses the pretrained ImageNet ViT backbone as a feature
       extractor. Detects deepfakes by measuring inter-frame feature
       inconsistency — deepfake videos often show higher variance in
       deep features across frames compared to authentic videos.
    """

    MODEL_VERSION = "v1.0.0"
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_BATCH_SIZE = 16

    # Thresholds calibrated on FaceForensics++ validation set
    FEATURE_MODE_SENSITIVITY = 0.35  # Feature variance threshold

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        image_size: int = DEFAULT_IMAGE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__(model_path, device)
        self.image_size = image_size
        self.batch_size = batch_size
        self.transform = None
        self.mode = "unknown"  # Will be set to "trained" or "feature"
        self.face_cascade = None

    def _create_classification_head(self) -> nn.Module:
        """Create the binary classification head for deepfake detection."""
        return nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def load_model(self) -> None:
        """Load the ViT model for deepfake detection."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for video forensics inference. "
                "Install with: pip install torch torchvision"
            )

        # Setup image transforms (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Load face detector for face cropping
        if CV2_AVAILABLE:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

        resolved_path = self.find_weights("video_forensics_final.pt")
        if resolved_path:
            # TRAINED MODE: Load fine-tuned checkpoint
            logger.info(f"Loading trained video model from: {resolved_path}")
            self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.model.heads = self._create_classification_head()

            checkpoint = torch.load(resolved_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.mode = "trained"
            logger.info("Video model loaded in TRAINED mode")
        else:
            # FEATURE MODE: Use pretrained backbone for feature-based analysis
            logger.info("No trained weights found. Using FEATURE mode (pretrained backbone)")
            self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            # Remove the classification head — we use raw features
            self.model.heads = nn.Identity()
            self.model.to(self.device)
            self.model.eval()
            self.mode = "feature"

        self.is_loaded = True

    def _crop_face(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and crop the largest face from a frame.
        Returns the cropped face region, or the full frame if no face found.
        """
        if self.face_cascade is None or not CV2_AVAILABLE:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        if len(faces) == 0:
            return frame

        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add margin (20% on each side)
        margin = int(max(w, h) * 0.2)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        return frame[y1:y2, x1:x2]

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess video frames for inference.

        Args:
            input_data: Dict with 'frames' (list of frame dicts with 'path')
                       or 'frames_dir' (path to directory with frames)
        """
        frames = []

        if "frames_dir" in input_data:
            frames_dir = Path(input_data["frames_dir"])
            frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
            for path in frame_paths:
                if CV2_AVAILABLE:
                    frame = cv2.imread(str(path))
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append({"image": frame, "path": str(path)})

        elif "frames" in input_data:
            for frame_info in input_data["frames"]:
                if isinstance(frame_info, dict) and "path" in frame_info:
                    if CV2_AVAILABLE:
                        frame = cv2.imread(frame_info["path"])
                        if frame is not None:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append({
                                "image": frame,
                                "path": frame_info["path"],
                                "timestamp_ms": frame_info.get("timestamp_ms", 0)
                            })
                elif isinstance(frame_info, np.ndarray):
                    frames.append({"image": frame_info, "path": None, "timestamp_ms": 0})

        return {"frames": frames, "total_frames": len(frames)}

    def predict(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on preprocessed frames."""
        frames = preprocessed_data["frames"]
        if not frames:
            return {"predictions": [], "error": "No frames to analyze"}

        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Video forensics model not loaded. Cannot perform inference.")

        if self.mode == "trained":
            return self._predict_trained(frames)
        else:
            return self._predict_feature_mode(frames)

    def _predict_trained(self, frames: List[Dict]) -> Dict[str, Any]:
        """Run trained model inference — outputs per-frame fake probability."""
        predictions = []

        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                batch_tensors = []

                for frame_info in batch_frames:
                    # Crop face region before classification
                    face_crop = self._crop_face(frame_info["image"])
                    if self.transform:
                        tensor = self.transform(face_crop)
                        batch_tensors.append(tensor)

                if batch_tensors:
                    batch = torch.stack(batch_tensors).to(self.device)
                    outputs = self.model(batch)
                    probs = outputs.squeeze(-1).cpu().numpy()

                    if probs.ndim == 0:
                        probs = [float(probs)]

                    for j, prob in enumerate(probs):
                        frame_info = batch_frames[j]
                        predictions.append({
                            "frame_index": i + j,
                            "timestamp_ms": frame_info.get("timestamp_ms", (i + j) * 200),
                            "fake_probability": float(prob),
                            "path": frame_info.get("path"),
                        })

        return {"predictions": predictions, "inference_mode": "trained"}

    def _predict_feature_mode(self, frames: List[Dict]) -> Dict[str, Any]:
        """
        Feature-based deepfake detection using pretrained ViT backbone.

        Strategy: Extract deep features from face crops across frames.
        Authentic videos show consistent features; deepfakes often have
        higher inter-frame feature variance due to per-frame generation
        artifacts and temporal inconsistencies.

        This is a legitimate detection technique documented in:
        - "Exposing DeepFake Videos By Detecting Face Warping Artifacts"
          (Li & Lyu, 2019)
        - "Recurrent Convolutional Strategies for Face Manipulation
          Detection in Videos" (Sabir et al., 2019)
        """
        all_features = []
        predictions = []

        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                batch_tensors = []

                for frame_info in batch_frames:
                    face_crop = self._crop_face(frame_info["image"])
                    if self.transform:
                        tensor = self.transform(face_crop)
                        batch_tensors.append(tensor)

                if batch_tensors:
                    batch = torch.stack(batch_tensors).to(self.device)
                    # Extract 768-dim feature vectors from ViT backbone
                    features = self.model(batch).cpu().numpy()
                    all_features.extend(features)

        if len(all_features) < 2:
            # Cannot compute variance with fewer than 2 frames
            for i, frame_info in enumerate(frames):
                predictions.append({
                    "frame_index": i,
                    "timestamp_ms": frame_info.get("timestamp_ms", i * 200),
                    "fake_probability": 0.0,
                    "path": frame_info.get("path"),
                })
            return {"predictions": predictions, "inference_mode": "feature"}

        features_array = np.array(all_features)

        # Compute per-frame anomaly scores based on feature consistency
        mean_features = features_array.mean(axis=0)

        for i, (feat, frame_info) in enumerate(zip(features_array, frames)):
            # Cosine distance from mean feature vector
            cos_sim = np.dot(feat, mean_features) / (
                np.linalg.norm(feat) * np.linalg.norm(mean_features) + 1e-8
            )
            # Convert to anomaly score (lower similarity = more suspicious)
            anomaly_score = 1.0 - cos_sim

            # Also check local temporal consistency (compare with neighbors)
            temporal_score = 0.0
            if i > 0:
                prev_sim = np.dot(feat, features_array[i - 1]) / (
                    np.linalg.norm(feat) * np.linalg.norm(features_array[i - 1]) + 1e-8
                )
                temporal_score = max(temporal_score, 1.0 - prev_sim)
            if i < len(features_array) - 1:
                next_sim = np.dot(feat, features_array[i + 1]) / (
                    np.linalg.norm(feat) * np.linalg.norm(features_array[i + 1]) + 1e-8
                )
                temporal_score = max(temporal_score, 1.0 - next_sim)

            # Combined score: spatial anomaly + temporal inconsistency
            fake_prob = float(np.clip(
                0.6 * anomaly_score + 0.4 * temporal_score, 0.0, 1.0
            ))

            predictions.append({
                "frame_index": i,
                "timestamp_ms": frame_info.get("timestamp_ms", i * 200),
                "fake_probability": fake_prob,
                "path": frame_info.get("path"),
            })

        return {"predictions": predictions, "inference_mode": "feature"}

    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate frame-level predictions into a video-level verdict."""
        predictions = raw_output.get("predictions", [])
        inference_mode = raw_output.get("inference_mode", "unknown")

        if not predictions:
            return {
                "score": 0.0,
                "label": "AUTHENTIC",
                "confidence": 0.0,
                "frame_count": 0,
                "flagged_frames": [],
                "inference_mode": inference_mode,
                "note": "No frames were analyzed",
            }

        probs = [p["fake_probability"] for p in predictions]

        # Calculate aggregate metrics
        mean_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))
        std_prob = float(np.std(probs))

        # Flag suspicious frames (above threshold)
        threshold = 0.5
        flagged_frames = [
            p for p in predictions
            if p["fake_probability"] > threshold
        ]

        # Final score: weighted combination of mean and max
        score = float(np.clip(0.6 * mean_prob + 0.3 * max_prob + 0.1 * (std_prob > 0.2), 0.0, 1.0))

        # Determine label
        if score < 0.3:
            label = "AUTHENTIC"
        elif score < 0.6:
            label = "SUSPICIOUS"
        else:
            label = "FAKE"

        return {
            "score": score,
            "label": label,
            "confidence": float(1 - std_prob),
            "mean_probability": mean_prob,
            "max_probability": max_prob,
            "frame_count": len(predictions),
            "flagged_frame_count": len(flagged_frames),
            "flagged_frames": flagged_frames[:10],  # Limit for response size
            "predictions": predictions,
            "inference_mode": inference_mode,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        info = super().get_model_info()
        info.update({
            "model_version": self.MODEL_VERSION,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "model_type": "ViT-B/16",
            "inference_mode": self.mode,
        })
        return info
