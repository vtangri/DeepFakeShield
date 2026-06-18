"""
Training script for audio spoof detection model.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchaudio
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import DeepfakeAudioDataset, create_data_loaders


class AudioSpoofModel(nn.Module):
    """Simple CNN for audio spoof detection."""
    
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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for waveform, labels in tqdm(loader, desc="Training"):
        waveform = waveform.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(waveform)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for waveform, labels in tqdm(loader, desc="Validating"):
            waveform = waveform.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(waveform)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train audio spoof detector")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    # Data
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir, batch_size=args.batch_size, modality="audio"
    )
    
    # Model
    model = AudioSpoofModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    writer = SummaryWriter(output_dir / "logs")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        
        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "audio_spoof_best.pt")
    
    # Test
    model.load_state_dict(torch.load(output_dir / "audio_spoof_best.pt"))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    torch.save(model.state_dict(), output_dir / "audio_spoof_final.pt")
    writer.close()


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch and torchaudio required.")
        exit(1)
    main()
