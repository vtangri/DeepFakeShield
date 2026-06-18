import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForensicTrainer:
    """
    Training pipeline for DeepFakeShield forensic models.
    Supports fine-tuning on large-scale datasets.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.BCELoss()
        logger.info(f"Trainer initialized on {device}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                
        return total_loss / len(dataloader)

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model checkpoint saved to {path}")

    def evaluate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        logger.info(f"Evaluation Accuracy: {accuracy:.2%}")
        return {"accuracy": accuracy}
