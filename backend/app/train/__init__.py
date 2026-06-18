"""
Training module for DeepFakeShield.
"""
from .dataset_manager import DatasetManager
from .trainer import ForensicTrainer
from .augmentation import DeepfakeAugmentation

__all__ = ["DatasetManager", "ForensicTrainer", "DeepfakeAugmentation"]
