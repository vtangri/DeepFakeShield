"""
Training script for video deepfake detection model.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import transforms
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from tqdm import tqdm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torchvision")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import DeepfakeVideoDataset, create_data_loaders


def create_model(num_classes: int = 1, pretrained: bool = True):
    """Create ViT model for binary classification."""
    if pretrained:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    else:
        model = vit_b_16(weights=None)
    
    # Replace classification head
    model.heads = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
        nn.Sigmoid() if num_classes == 1 else nn.Identity()
    )
    
    return model


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (frames, labels) in enumerate(pbar):
        # frames: (batch, num_frames, C, H, W) - use middle frame
        if frames.dim() == 5:
            frames = frames[:, frames.shape[1] // 2]
        
        frames = frames.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Validating"):
            if frames.dim() == 5:
                frames = frames[:, frames.shape[1] // 2]
            
            frames = frames.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train video deepfake detector")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        modality="video"
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Model
    model = create_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # Tensorboard
    writer = SummaryWriter(output_dir / "logs")
    
    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, output_dir / "best_model.pt")
            print(f"Saved best model with val_acc: {val_acc:.4f}")
    
    # Test
    print("\nTesting best model...")
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, preds, labels = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save final
    torch.save(model.state_dict(), output_dir / "video_forensics_final.pt")
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc,
            "epochs": args.epochs,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    writer.close()
    print(f"\nTraining complete. Model saved to {output_dir}")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch is required for training.")
        exit(1)
    main()
