"""
Module containing training and evaluation routines.
"""

import os
from typing import List
import torch
from torch import nn, optim
# Instead of importing from torch.cuda.amp, we now use torch.amp directly.
from torch.utils.data import DataLoader, Dataset

class ExampleDataset(Dataset):
    """
    A simple dataset for demonstration purposes.
    Replace with your actual dataset when available.
    """
    def __init__(self, data: List[torch.Tensor], labels: List[torch.Tensor]) -> None:
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]

def train_model(model, dataloader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device,
                scaler: torch.amp.GradScaler, num_epochs: int = 20,
                scheduler: optim.lr_scheduler._LRScheduler = None) -> None:
    """
    Train the model for a given number of epochs.
    Supports both:
      - Old format: (inputs, labels)
      - New format: (guess_indices, constraint_indices, presence_list, absent_list, labels)
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Old format: inputs, labels
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
            elif isinstance(batch, (list, tuple)) and len(batch) == 5:
                # New format: guess_indices, constraint_indices, presence_list, absent_list, labels
                guess_indices, constraint_indices, presence_list, absent_list, labels = batch
                guess_indices = guess_indices.to(device, non_blocking=True)
                constraint_indices = constraint_indices.to(device, non_blocking=True)
                # presence_list and absent_list remain as lists; the model handles them internally.
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(guess_indices, constraint_indices, presence_list, absent_list)
            else:
                raise ValueError("Unexpected batch format from dataloader.")

            # Convert outputs to float32 to match labels for loss computation.
            loss = criterion(outputs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size

        epoch_loss /= len(dataloader.dataset)
        print(f"[INFO] Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f}")

def evaluate_model(model, dataloader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> float:
    """
    Evaluate the model on a validation set.
    Supports both input formats.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
            elif isinstance(batch, (list, tuple)) and len(batch) == 5:
                guess_indices, constraint_indices, presence_list, absent_list, labels = batch
                guess_indices = guess_indices.to(device, non_blocking=True)
                constraint_indices = constraint_indices.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(guess_indices, constraint_indices, presence_list, absent_list)
            else:
                raise ValueError("Unexpected batch format from dataloader.")

            loss = criterion(outputs.float(), labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
    total_loss /= len(dataloader.dataset)
    print(f"[INFO] Evaluation Loss: {total_loss:.4f}")
    return total_loss

def demo_training() -> None:
    """
    A demonstration training routine using ExampleDataset.
    This uses the old two-element format for demonstration.
    """
    # Configuration parameters for dummy data.
    input_size = 100  # Adjust to your actual input dimension.
    hidden_size = 1024
    num_hidden_layers = 3
    dropout = 0.3
    batch_size = 256
    learning_rate = 1e-3
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy training and validation data.
    train_data = [torch.randn(input_size) for _ in range(2000)]
    train_labels = [torch.randn(1) for _ in range(2000)]
    val_data = [torch.randn(input_size) for _ in range(200)]
    val_labels = [torch.randn(1) for _ in range(200)]

    train_dataset = ExampleDataset(train_data, train_labels)
    val_dataset = ExampleDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # For demo purposes, we use the old model.
    from model import HeuristicScoringModel
    model = HeuristicScoringModel(input_size, hidden_size, num_hidden_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
            else:
                raise ValueError("Demo training expects old format (inputs, labels).")

            loss = criterion(outputs.float(), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * labels.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"[INFO] Epoch {epoch + 1}/{num_epochs} | Training Loss: {epoch_loss:.4f}")

        # Evaluate after each epoch.
        val_loss = evaluate_model(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"[INFO] Saved best model at epoch {epoch + 1}")
        scheduler.step()

if __name__ == "__main__":
    demo_training()
