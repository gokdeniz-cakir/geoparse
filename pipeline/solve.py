
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass

from parser.model import DiagramParser
from solver.model import NeuralSolver
from solver.refinement import refine_coordinates

@dataclass
class GeometricSolution:
    raw_coords: np.ndarray
    refined_coords: np.ndarray
    angles: list[float]
    sides: list[float]
    area: float
    perplexity: float # heuristic confidence?

class GeometrySolver:
    def __init__(self, parser_path, solver_path, device="cpu"):
        self.device = device
        
        # Load Parser
        self.parser = DiagramParser(output_dim=8).to(device)
        self.parser.load_state_dict(torch.load(parser_path, map_location=device)["model_state_dict"])
        self.parser.eval()
        
        # Load Solver
        self.solver = NeuralSolver(output_dim=6).to(device)
        self.solver.load_state_dict(torch.load(solver_path, map_location=device)["model_state_dict"])
        self.solver.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Vocab map (simplified)
        self.PROBLEM_TYPES = [
            "right_triangle", "isosceles_triangle", "equilateral_triangle", 
            "angle_chase", "law_of_cosines"
        ]

    def solve(self, image_path, problem_type="right_triangle"):
        """
        End-to-end solve.
        image_path: params
        problem_type: Used for Refinement constraints (e.g. 'right_triangle')
        """
        # 1. Vision: Parse Coordinates
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            tensor_coords = self.parser(input_tensor)
            
        # 2. Refinement: Snap to constraints
        # Move to CPU for optimization loop if needed, or keep on device
        # Refinement uses torch optimization
        tensor_coords_refined = refine_coordinates(tensor_coords.cpu(), problem_type).to(self.device)
        
        # 3. Reasoning: Predict Properties
        # Solver needs type_idx (we use 0 as established)
        type_idx = torch.tensor([0], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            props = self.solver((tensor_coords_refined, type_idx))[0]
            
        # 4. Decode
        p = props.cpu().numpy()
        angles = (p[:3] * 180.0).tolist()
        sides = (p[3:] * 20.0).tolist()
        
        # Calculate Area (Heron's Formula or 0.5*a*b*sinC)
        # Using Heron's for stability with side predictions
        s1, s2, s3 = sides[:3]
        s = (s1 + s2 + s3) / 2
        area = np.sqrt(max(0, s * (s - s1) * (s - s2) * (s - s3)))
        
        return GeometricSolution(
            raw_coords=tensor_coords.cpu().numpy()[0] * 20.0,
            refined_coords=tensor_coords_refined.cpu().numpy()[0] * 20.0,
            angles=angles,
            sides=sides,
            area=float(area),
            perplexity=0.0 # Placeholder
        )

if __name__ == "__main__":
    # Test run
    solver = GeometrySolver(
        "experiments/parser_v4/best_model.pt", 
        "experiments/solver_baseline/best_solver.pt"
    )
    print("Pipeline Initialized.")
