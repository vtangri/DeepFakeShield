"""
Unit tests for ML inference services.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml"))

from inference.base import BaseInferenceService, EnsembleService
from inference.video_forensics import VideoForensicsService
from inference.audio_spoof import AudioSpoofService
from inference.lipsync import LipSyncService
from inference.fusion import MultimodalFusionService


class TestVideoForensicsService:
    """Test video forensics inference."""
    
    @pytest.fixture
    def service(self):
        return VideoForensicsService(device="cpu")
    
    def test_initialization(self, service):
        assert service.device == "cpu"
        assert service.image_size == 224
        assert not service.is_loaded
    
    def test_load_model(self, service):
        service.load_model()
        assert service.is_loaded
    
    def test_preprocess_empty_input(self, service):
        result = service.preprocess({"frames": []})
        assert result["frames"] == []
        assert result["total_frames"] == 0
    
    def test_postprocess_empty(self, service):
        result = service.postprocess({"predictions": []})
        assert result["score"] == 0.0
        assert result["label"] == "AUTHENTIC"
    
    def test_postprocess_with_predictions(self, service):
        predictions = [
            {"frame_index": i, "fake_probability": 0.1, "timestamp_ms": i * 200}
            for i in range(10)
        ]
        result = service.postprocess({"predictions": predictions})
        assert 0 <= result["score"] <= 1
        assert result["frame_count"] == 10
        assert result["label"] in ["AUTHENTIC", "SUSPICIOUS", "FAKE"]
    
    def test_model_info(self, service):
        info = service.get_model_info()
        assert "model_version" in info
        assert "image_size" in info


class TestAudioSpoofService:
    """Test audio spoof detection."""
    
    @pytest.fixture
    def service(self):
        return AudioSpoofService(device="cpu")
    
    def test_initialization(self, service):
        assert service.sample_rate == 16000
    
    def test_load_model(self, service):
        service.load_model()
        assert service.is_loaded
    
    def test_preprocess_missing_file(self, service):
        result = service.preprocess({"audio_path": "/nonexistent/audio.wav"})
        assert "error" in result or result.get("waveform") is None
    
    def test_postprocess(self, service):
        result = service.postprocess({
            "spoof_probability": 0.2,
            "duration_sec": 5.0
        })
        assert result["label"] == "AUTHENTIC"
        assert result["score"] == 0.2


class TestLipSyncService:
    """Test lip-sync verification."""
    
    @pytest.fixture
    def service(self):
        return LipSyncService(device="cpu")
    
    def test_initialization(self, service):
        assert service.window_size_ms == 500
    
    def test_preprocess_empty(self, service):
        result = service.preprocess({
            "frames": {"frames": []},
            "transcript": {}
        })
        assert result["mouth_features"] == []
    
    def test_postprocess_synchronized(self, service):
        result = service.postprocess({
            "mismatch_score": 0.1,
            "segments": []
        })
        assert result["label"] == "SYNCHRONIZED"


class TestMultimodalFusionService:
    """Test multimodal fusion."""
    
    @pytest.fixture
    def service(self):
        return MultimodalFusionService()
    
    def test_default_weights(self, service):
        assert service.weights["video"] == 0.45
        assert service.weights["audio"] == 0.30
        assert service.weights["lipsync"] == 0.25
    
    def test_fusion_authentic(self, service):
        service.load_model()
        result = service({
            "video": {"score": 0.1, "confidence": 0.9},
            "audio": {"score": 0.1, "confidence": 0.9},
            "lipsync": {"score": 0.1, "confidence": 0.9},
        })
        assert result["label"] == "AUTHENTIC"
        assert result["overall_score"] < 0.3
    
    def test_fusion_fake(self, service):
        service.load_model()
        result = service({
            "video": {"score": 0.9, "confidence": 0.9},
            "audio": {"score": 0.8, "confidence": 0.9},
            "lipsync": {"score": 0.85, "confidence": 0.9},
        })
        assert result["label"] in ["LIKELY_FAKE", "FAKE"]
        assert result["overall_score"] > 0.7


class TestEnsembleService:
    """Test ensemble of services."""
    
    def test_ensemble_averages_scores(self):
        # Create mock services
        service1 = MagicMock()
        service1.return_value = {"score": 0.2}
        
        service2 = MagicMock()
        service2.return_value = {"score": 0.4}
        
        ensemble = EnsembleService([service1, service2])
        result = ensemble("test_input")
        
        assert pytest.approx(result["ensemble_score"]) == 0.3  # (0.2 + 0.4) / 2
        assert len(result["individual_results"]) == 2
