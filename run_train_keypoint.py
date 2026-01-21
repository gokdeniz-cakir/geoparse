"""
Training script for Keypoint Detection Model.

Trains the heatmap-based vertex detector on synthetic geometry diagrams.
Optimized with Mixed Precision Training (AMP) and parallel data loading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
import os

from parser.keypoint_model import KeypointDetector
from parser.keypoint_dataset import KeypointDataset


def train_keypoint_model(
    data_dir: str = "data/dataset",
    output_dir: str = "experiments/keypoint_v1",
    epochs: int = 20,
    batch_size: int = 64,  # Increased from 32
    lr: float = 1e-4,
    device: str = "mps",  # Use MPS on Mac, "cuda" on NVIDIA, "cpu" fallback
    num_workers: int = 8,  # Increased parallelization
    use_amp: bool = True  # Mixed precision training
):
    """Train the keypoint detection model with optimized parallelization."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check device availability
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}, AMP: {use_amp}")
    
    # Datasets
    train_dataset = KeypointDataset(data_dir, split="train")
    val_dataset = KeypointDataset(data_dir, split="val")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Optimized DataLoaders with parallelization
    loader_kwargs = {
        'num_workers': num_workers,
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        **loader_kwargs
    )
    
    # Model
    model = KeypointDetector(num_keypoints=4, heatmap_size=56)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Mixed precision scaler (for CUDA; MPS has limited AMP support)
    scaler = GradScaler() if device == "cuda" and use_amp else None
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            images = batch['image'].to(device)
            targets = batch['heatmap'].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler is not None:
                with autocast(device_type=device, dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = batch['heatmap'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_path / "best_model.pt")
            print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")
            
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_path / 'best_model.pt'}")


if __name__ == "__main__":
    train_keypoint_model(
        data_dir="data/dataset",
        output_dir="experiments/keypoint_v1",
        epochs=20,
        batch_size=64,
        lr=1e-4,
        num_workers=8
    )
