
import torch
from pathlib import Path
from solver.model import NeuralSolver
from solver.dataset import get_solver_dataloaders

def evaluate_solver(model_path="experiments/solver_baseline/best_solver.pt"):
    device = "cpu"
    
    # Load Model with output_dim=6 to match Property Regressor
    model = NeuralSolver(output_dim=6)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint['epoch']} (Loss: {checkpoint['loss']:.4f})")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Assuming we might want to run even if load fails for debugging, or return
        # But usually we want to return if no model
        # For now, let's just print and continue with random weights to show format
        
    model.eval()
    
    # Load Data
    _, val_dl = get_solver_dataloaders("data/dataset", batch_size=32)
    
    total_ae = 0.0
    count = 0
    
    print("\nSample Predictions (Vector: [A1, A2, A3, S1, S2, S3])")
    print("Top row: Pred | Bottom row: Actual")
    print("-" * 50)
    
    with torch.no_grad():
        for i, (inputs, y) in enumerate(val_dl):
            # inputs is (x, type_idx)
            pred = model(inputs)
            
            # Calculate Absolute Error
            abs_err = torch.abs(pred - y)
            total_ae += torch.sum(abs_err).item()
            count += len(y) * 6 # 6 dims
            
            # Print first sample of first batch
            if i == 0:
                for j in range(min(3, len(y))):
                    p = pred[j].numpy()
                    a = y[j].numpy()
                    print(f"Sample {j}:")
                    print(f"  P: [{', '.join([f'{v:.2f}' for v in p])}]")
                    print(f"  A: [{', '.join([f'{v:.2f}' for v in a])}]")
                    print("-" * 20)
    
    mae = total_ae / count
    print("-" * 36)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

if __name__ == "__main__":
    evaluate_solver()
