
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from solver.model import NeuralSolver
from solver.dataset import get_solver_dataloaders

def train_solver(
    data_dir="data/dataset",
    output_dir="experiments/solver_baseline",
    epochs=50,
    batch_size=64,
    lr=1e-3
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data
    train_dl, val_dl = get_solver_dataloaders(data_dir, batch_size=batch_size)
    
    # Model
    model = NeuralSolver(output_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    # Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, y in pbar:
            # inputs is (x, type_idx)
            x, type_idx = inputs
            x = x.to(device)
            type_idx = type_idx.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            # Pass tuple to model
            pred = model((x, type_idx))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_dl)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, y in val_dl:
                x, type_idx = inputs
                x = x.to(device)
                type_idx = type_idx.to(device)
                y = y.to(device)
                
                pred = model((x, type_idx))
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_dl)
        
        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, output_path / "best_solver.pt")
            print(f"  -> Saved best model ({best_loss:.4f})")
            
    print("Solver Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/dataset")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_solver(data_dir=args.data_dir, epochs=args.epochs)
