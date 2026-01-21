
import torch
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms
from parser.model import DiagramParser
from parser.dataset import DiagramDataset

def debug_scale():
    # Load dataset to get a real sample with ground truth
    data_dir = Path("data/dataset")
    dataset = DiagramDataset(data_dir, split="train", max_samples=10)
    
    # Get first sample
    idx = 0
    image, target, answer = dataset[idx] # target is tensor of 8 floats
    item = dataset.metadata[idx]
    
    print(f"--- Sample {idx} ---")
    print(f"Goal: {item['goal']}")
    print(f"Original Vertices (Ground Truth in structure dict):")
    # Sort them as the dataset does to match
    vertices = item["structure"]["vertices"]
    sorted_v = sorted(vertices.items(), key=lambda v: (-v[1][1], v[1][0]))
    for k, v in sorted_v[:4]:
        print(f"  {k}: {v}")
        
    print(f"\nTarget Tensor (Normalized for training):")
    print(target)
    
    # Check normalization factor derived from target
    # We expect target * 20.0 approx equals Original (padded)
    print(f"Target * 20.0:")
    print(target * 20.0)
    
    # Run Model
    device = "cpu"
    model_path = "experiments/parser_v4/best_model.pt"
    
    model = DiagramParser(pretrained=False, output_dim=8)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    input_tensor = image.unsqueeze(0) # Add batch dim
    
    with torch.no_grad():
        output = model(input_tensor)
        
    raw_pred = output[0]
    print(f"\nRaw Model Output:")
    print(raw_pred)
    
    print(f"\nDenormalized Output (* 20.0):")
    print(raw_pred * 20.0)
    
    # Compare
    mse = torch.mean((raw_pred - target)**2)
    print(f"\nMSE (Raw): {mse.item()}")
    
    mse_denorm = torch.mean(((raw_pred * 20.0) - (target * 20.0))**2)
    print(f"MSE (Denorm): {mse_denorm.item()}")

if __name__ == "__main__":
    debug_scale()
