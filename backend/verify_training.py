import torch
import torch.nn as nn
from app.train import DatasetManager, ForensicTrainer, DeepfakeAugmentation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Mock Model for verification
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 222 * 222, 1)  # Assuming 224x224 input
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)

def run_dry_run():
    logger.info("--- Starting Training Infrastructure Dry-Run ---")
    
    # 1. Test Dataset Manager
    dm = DatasetManager(base_path="data/test_datasets")
    available = dm.list_available_datasets()
    logger.info(f"Available datasets: {list(available.keys())}")
    
    path = dm.prepare_dataset("celeb_df")
    logger.info(f"Dataset path prepared: {path}")

    # 2. Test Augmentation
    aug = DeepfakeAugmentation()
    mock_img = (torch.rand(224, 224, 3).numpy() * 255).astype('uint8')
    augmented = aug.transform(mock_img)
    logger.info(f"Augmentation successful. Shape: {augmented.shape}")

    # 3. Test Trainer
    model = MockModel()
    trainer = ForensicTrainer(model)
    
    # Simulate a single batch
    mock_data = torch.rand(2, 3, 224, 224)
    mock_target = torch.tensor([[0.0], [1.0]])
    
    logger.info("Running dummy training step...")
    trainer.optimizer.zero_grad()
    output = model(mock_data)
    loss = trainer.criterion(output, mock_target)
    loss.backward()
    trainer.optimizer.step()
    
    logger.info(f"Training step successful. Loss: {loss.item():.4f}")
    logger.info("--- Dry-Run Completed Successfully ---")

if __name__ == "__main__":
    run_dry_run()
