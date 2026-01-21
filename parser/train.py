"""
Training script for the diagram parser.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .dataset import get_dataloaders
from .model import DiagramParser


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for images, targets, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    device: str = "auto"
):
    """
    Train the diagram parser model.
    
    Args:
        data_dir: Path to synthetic data directory
        output_dir: Path to save model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        num_workers: Number of dataloader workers for parallel loading
        device: Device to train on ("auto", "cuda", "mps", "cpu")
    """
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (with parallel workers for speed)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers, max_samples=max_samples
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    model = DiagramParser(pretrained=True, freeze_backbone=False)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, output_dir / "best_model.pt")
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt")["model_state_dict"])
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/parser_v1"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )
