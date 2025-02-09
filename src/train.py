# src/train.py
import torch
from torch.utils.data import DataLoader, Dataset
from model import HeuristicScoringModel
from typing import List
from torch import nn, optim
from torch.cuda import amp
import os


# Example Dataset (Replace with your actual dataset)
class ExampleDataset(Dataset):
    def __init__(self, data: List[torch.Tensor], labels: List[torch.Tensor]):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_model(
        model: HeuristicScoringModel,
        dataloader: DataLoader,
        criterion,
        optimizer,
        device: torch.device,
        scaler: amp.GradScaler,
        num_epochs: int = 20,
        scheduler=None
):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(dataloader.dataset)
        print(f"[INFO] Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def evaluate_model(
        model: HeuristicScoringModel,
        dataloader: DataLoader,
        criterion,
        device: torch.device
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

    total_loss /= len(dataloader.dataset)
    print(f"[INFO] Evaluation Loss: {total_loss:.4f}")
    return total_loss


def main():
    # Configuration
    input_size = 100  # Replace with your actual input size
    hidden_size = 1024
    num_hidden_layers = 3
    dropout = 0.3
    batch_size = 64
    learning_rate = 1e-3
    num_epochs: int = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Dataset and DataLoader (Replace with your actual data)
    # For demonstration, using random tensors
    train_data = [torch.randn(input_size) for _ in range(2000)]
    train_labels = [torch.randn(1) for _ in range(2000)]
    val_data = [torch.randn(input_size) for _ in range(200)]
    val_labels = [torch.randn(1) for _ in range(200)]

    train_dataset = ExampleDataset(train_data, train_labels)
    val_dataset = ExampleDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model, Criterion, Optimizer, and Scaler
    model = HeuristicScoringModel(input_size, hidden_size, num_hidden_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = amp.GradScaler()

    # (Optional) Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"[INFO] Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Evaluation after each epoch
        val_loss = evaluate_model(model, val_loader, criterion, device)

        # (Optional) Save the best model
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"[INFO] Saved Best Model at Epoch {epoch + 1}")


if __name__ == "__main__":
    main()
