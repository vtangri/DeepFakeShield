"""
Dataset utilities for deepfake detection training.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DeepfakeVideoDataset(Dataset):
    """Dataset for video deepfake detection training."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        frames_per_video: int = 16,
        image_size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load sample list from manifest or directory structure."""
        samples = []
        
        # Try loading from manifest
        manifest_path = self.data_dir / f"{self.split}.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                samples = json.load(f)
        else:
            # Scan directory structure
            # Expected: data_dir/split/real/*.mp4, data_dir/split/fake/*.mp4
            split_dir = self.data_dir / self.split
            
            for label, subdir in [(0, "real"), (1, "fake")]:
                subdir_path = split_dir / subdir
                if subdir_path.exists():
                    for video_path in subdir_path.glob("*.mp4"):
                        samples.append({
                            "path": str(video_path),
                            "label": label,
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        video_path = sample["path"]
        label = sample["label"]
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        else:
            frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        return frames, torch.tensor(label, dtype=torch.float32)
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract evenly spaced frames from video."""
        frames = []
        
        if not CV2_AVAILABLE:
            # Return dummy frames
            return [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8) 
                    for _ in range(self.frames_per_video)]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8) 
                    for _ in range(self.frames_per_video)]
        
        # Sample evenly
        indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        
        cap.release()
        return frames


class DeepfakeAudioDataset(Dataset):
    """Dataset for audio spoof detection training."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration_sec: float = 4.0,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load audio samples."""
        samples = []
        manifest_path = self.data_dir / f"{self.split}.json"
        
        if manifest_path.exists():
            with open(manifest_path) as f:
                samples = json.load(f)
        else:
            split_dir = self.data_dir / self.split
            for label, subdir in [(0, "bonafide"), (1, "spoof")]:
                subdir_path = split_dir / subdir
                if subdir_path.exists():
                    for audio_path in subdir_path.glob("*.wav"):
                        samples.append({"path": str(audio_path), "label": label})
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        audio_path = sample["path"]
        label = sample["label"]
        
        # Load audio
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Pad or truncate
            if waveform.shape[1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            elif waveform.shape[1] < self.max_samples:
                padding = self.max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        except Exception:
            waveform = torch.zeros(1, self.max_samples)
        
        return waveform.squeeze(0), torch.tensor(label, dtype=torch.float32)


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    modality: str = "video",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    if modality == "video":
        DatasetClass = DeepfakeVideoDataset
    else:
        DatasetClass = DeepfakeAudioDataset
    
    train_dataset = DatasetClass(data_dir, split="train")
    val_dataset = DatasetClass(data_dir, split="val")
    test_dataset = DatasetClass(data_dir, split="test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader
