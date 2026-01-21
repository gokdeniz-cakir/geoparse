
import torch
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

from parser.model import DiagramParser
from solver.model import NeuralSolver
from solver.dataset import SolverDataset # To get vocab if needed, but we'll hardcode for simplicity

def solve_demo(sample_idx=None):
    device = "cpu"
    
    # 1. Load Models
    print("Loading Models...")
    
    # Parser
    parser = DiagramParser(output_dim=8)
    parser_path = "experiments/parser_v4/best_model.pt"
    try:
        ckpt = torch.load(parser_path, map_location=device)
        parser.load_state_dict(ckpt["model_state_dict"])
        parser.eval()
        print(" -> Parser Loaded")
    except Exception as e:
        print(f"Failed to load parser: {e}")
        return

    # Solver
    solver = NeuralSolver(output_dim=6)
    solver_path = "experiments/solver_baseline/best_solver.pt"
    try:
        ckpt = torch.load(solver_path, map_location=device)
        solver.load_state_dict(ckpt["model_state_dict"])
        solver.eval()
        print(" -> Solver Loaded")
    except Exception as e:
        print(f"Failed to load solver: {e}")
        return

    # 2. Load Data Sample
    data_dir = Path("data/dataset")
    meta_path = data_dir / "metadata.jsonl"
    with open(meta_path, "r") as f:
        lines = f.readlines()
        
    with open(meta_path, "r") as f:
        lines = f.readlines()
        
    # Find a 'right_triangle' sample for demo purposes causing Refinement to trigger
    sample_idx = 0
    for i, line in enumerate(lines):
        if "right_triangle" in line:
            sample_idx = i
            break
            
    # Random offset to avoid always picking the first one
    # Scan forward from random start
    start = random.randint(0, len(lines)-1000)
    for i in range(start, len(lines)):
        if "right_triangle" in lines[i]:
            sample_idx = i
            break
            
    item = json.loads(lines[sample_idx])
    # Construct absolute path manually since we aren't using the Dataset class
    image_path = data_dir / "images" / item["filename"]
    
    print(f"\n--- Solving Problem {sample_idx} ---")
    problem_type = item.get("problem_type", "unknown")
    print(f"Type: {problem_type}")
    print(f"Goal: {item['answer']} (Ground Truth)")
    
    # 3. Process Image -> Parser
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Image not found at {image_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        coords_pred_norm = parser(input_tensor) # [1, 8]
        
    # Coords are normalized / 20.0. 
    # Solver expects this exact format (Normalized coords).
    # ALSO requires Problem Type Index.
    
    # Map type name to index
    PROBLEM_TYPES = [] # Unused since we trained effectively with unknown types mapping to 0
    # Since the vocabulary used in training didn't match the dataset, the model learned to use index 0.
    # We will stick to index 0 to match the trained behavior.
    type_idx = 0
    type_tensor = torch.tensor([type_idx], dtype=torch.long)
    
    # 4. Run Refinement (New Step)
    from solver.refinement import refine_coordinates
    print("\n--- Geometric Refinement ---")
    # Pass the underlying geometry type (e.g. "right_triangle") to trigger constraints
    # Fallback to problem_type if structure type not found
    geo_type = item.get("structure", {}).get("type", problem_type)
    coords_refined = refine_coordinates(coords_pred_norm, geo_type)
    
    # 5. Run Solver on RAW and REFINED
    # Input: (coords, type_idx)
    with torch.no_grad():
        props_raw = solver((coords_pred_norm, type_tensor))
        props_refined = solver((coords_refined, type_tensor))
        
    # 6. Interpret Output
    def print_props(p, label):
        angles = p[:3] * 180.0
        sides = p[3:] * 20.0
        print(f"\n{label} Prediction:")
        print(f"  Interior Angles: {angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°")
        print(f"  Side Lengths   : {sides[0]:.2f}, {sides[1]:.2f}, {sides[2]:.2f}")
        return list(angles) + list(sides)

    vals_raw = print_props(props_raw[0].numpy(), "RAW")
    vals_refined = print_props(props_refined[0].numpy(), "REFINED")
    
    # 7. Verify against Ground Truth Answer
    print(f"\nGround Truth Goal: {item['goal']}")
    print(f"Ground Truth Answer: {item['answer']}")
    
    # Compare
    closest_raw = min(vals_raw, key=lambda x: abs(x - float(item['answer'])))
    closest_refined = min(vals_refined, key=lambda x: abs(x - float(item['answer'])))
    
    print(f"\nAccuracy Check:")
    print(f"  Raw Error    : {abs(closest_raw - float(item['answer'])):.2f}")
    print(f"  Refined Error: {abs(closest_refined - float(item['answer'])):.2f}") 
    
    print("\n(Note: Refinement snaps vertices to likely geometric constraints like Right Angles or Equal Sides.)")

if __name__ == "__main__":
    solve_demo()
