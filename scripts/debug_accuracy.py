
import torch
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from parser.model import DiagramParser
from solver.model import NeuralSolver

def debug_accuracy():
    device = "cpu"
    print("--- Debugging Accuracy Pipeline ---")

    # 1. Load Models
    parser = DiagramParser(output_dim=8)
    try:
        parser.load_state_dict(torch.load("experiments/parser_v4/best_model.pt", map_location=device)["model_state_dict"])
        parser.eval()
    except Exception as e:
        print(f"Parser load failed: {e}")
        return

    solver = NeuralSolver(output_dim=6)
    try:
        solver.load_state_dict(torch.load("experiments/solver_baseline/best_solver.pt", map_location=device)["model_state_dict"])
        solver.eval()
    except Exception as e:
        print(f"Solver load failed: {e}")
        return

    # 2. Get Data
    data_dir = Path("data/dataset")
    with open(data_dir / "metadata.jsonl", "r") as f:
        lines = f.readlines()
        
    # Analyze 5 random samples
    indices = random.sample(range(len(lines)), 5)
    
    # Vocabulary mapping (from training)
    PROBLEM_TYPES = [] # Unused, index 0
    
    for idx in indices:
        item = json.loads(lines[idx])
        print(f"\nSample {idx}: {item.get('problem_type', 'unknown')}")
        print(f"Goal: {item['answer']} ({item.get('given', [])} -> {item.get('hidden', [])})")
        
        # --- A. Ground Truth Coordinates ---
        # Sort vertices like training
        structure = item["structure"]
        sorted_v = sorted(structure["vertices"].items(), key=lambda v: (-v[1][1], v[1][0]))
        points = [p[1] for p in sorted_v[:4]]
        coords_gt = []
        for p in points: coords_gt.extend(p)
        if points:
            last = points[-1]
            while len(coords_gt) < 8: coords_gt.extend(last)
        else: coords_gt = [0]*8
            
        tensor_gt = torch.tensor(coords_gt).float().unsqueeze(0) / 20.0
        
        # --- B. Predicted Coordinates (Parser) ---
        img_path = data_dir / "images" / item["filename"]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Image not found")
            continue
            
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_img = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            tensor_pred = parser(input_img)
            
        # --- Error in Coordinates ---
        coord_mse = torch.mean((tensor_gt - tensor_pred)**2).item() * (20.0**2)
        print(f"  Parser MSE (Pixel Space): {coord_mse:.4f}")
        
        # --- C. Run Solver on Both ---
        type_tensor = torch.tensor([0], dtype=torch.long)
        
        with torch.no_grad():
            out_gt = solver((tensor_gt, type_tensor))[0] # [6]
            out_pred = solver((tensor_pred, type_tensor))[0] # [6]
            
        # Interpret (Angles, Sides)
        def fmt(t):
            ang = t[:3] * 180.0
            sid = t[3:] * 20.0
            return f"A=[{ang[0]:.0f}, {ang[1]:.0f}, {ang[2]:.0f}], S=[{sid[0]:.1f}, {sid[1]:.1f}, {sid[2]:.1f}]"
            
        print(f"  Solver(GT Coords):   {fmt(out_gt)}")
        print(f"  Solver(Pred Coords): {fmt(out_pred)}")
        
        # Compare Side 1 (usually S12, the first side)
        s1_gt = out_gt[3].item() * 20.0
        s1_pred = out_pred[3].item() * 20.0
        err = abs(s1_gt - s1_pred)
        print(f"  Side 1 Error: {err:.2f} ({(err/s1_gt)*100:.1f}%)")

if __name__ == "__main__":
    debug_accuracy()
