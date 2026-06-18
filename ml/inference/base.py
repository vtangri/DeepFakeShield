"""
Base Inference Service interface.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


class BaseInferenceService(ABC):
    """Abstract base class for all inference services."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights."""
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Preprocess input data for inference."""
        pass
    
    @abstractmethod
    def predict(self, preprocessed_data: Any) -> Dict[str, Any]:
        """Run inference on preprocessed data."""
        pass
    
    @abstractmethod
    def postprocess(self, raw_output: Any) -> Dict[str, Any]:
        """Postprocess model output."""
        pass
    
    def __call__(self, input_data: Any) -> Dict[str, Any]:
        """Run full inference pipeline."""
        if not self.is_loaded:
            self.load_model()
        
        preprocessed = self.preprocess(input_data)
        raw_output = self.predict(preprocessed)
        return self.postprocess(raw_output)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded,
        }


class EnsembleService:
    """Ensemble multiple inference services."""
    
    def __init__(self, services: List[BaseInferenceService], weights: Optional[List[float]] = None):
        self.services = services
        self.weights = weights or [1.0 / len(services)] * len(services)
    
    def __call__(self, input_data: Any) -> Dict[str, Any]:
        """Run ensemble inference."""
        results = []
        for service in self.services:
            try:
                result = service(input_data)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "score": 0.5})
        
        # Weighted average of scores
        weighted_score = sum(
            r.get("score", 0.5) * w 
            for r, w in zip(results, self.weights)
        )
        
        return {
            "ensemble_score": weighted_score,
            "individual_results": results,
        }
