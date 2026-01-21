
import torch
import json
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class SolverDataset(Dataset):
    """
    Dataset for the Neural Solver.
    Input: Normalized vertex coordinates (from metadata).
    Output: Numerical answer.
    """
    def __init__(self, data_dir, split="train", max_samples=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.items = []
        
        # Hardcoded list of problem types (from generator) for vocabulary
        self.PROBLEM_TYPES = [
            "right_triangle_345", "right_triangle_5_12_13", 
            "right_triangle_angle", "right_triangle_hypotenuse",
            "isosceles_triangle_base_angles", "isosceles_triangle_sides", "isosceles_triangle_apex",
            "equilateral_triangle_area", "equilateral_triangle_height", "equilateral_triangle_perimeter",
            "angle_sum_alpha", "angle_sum_two_known",
            "external_angle_interior", "external_angle_exterior",
            "find_altitude", "find_BD", "find_DC",
            "law_of_cosines_find_side", "law_of_cosines_find_angle",
            "cevian_angle_chase"
        ]
        self.type_to_idx = {t: i for i, t in enumerate(self.PROBLEM_TYPES)}
        
        # Load metadata
        meta_path = self.data_dir / "metadata.jsonl"
        
        with open(meta_path, "r") as f:
            lines = f.readlines()
            
        total = len(lines)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        
        if split == "train":
            lines = lines[:train_size]
        elif split == "val":
            lines = lines[train_size:train_size+val_size]
        else: # test
            lines = lines[train_size+val_size:]
            
        if max_samples:
            lines = lines[:max_samples]
            
        for line in lines:
            self.items.append(json.loads(line))
            
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        structure = item["structure"]
        
        # Problem Type Encoding
        p_type = item.get("problem_type_name", "unknown")
        # Handle cases where type name might differ slightly or be missing
        # For now assume it matches or map to 0
        type_idx = self.type_to_idx.get(p_type, 0)
        
        # --- INPUT FEATURE ENGINEERING ---
        vertices = structure["vertices"]
        sorted_vertices = sorted(
            vertices.items(), 
            key=lambda v: (-v[1][1], v[1][0])
        )
        
        coords = []
        points_list = [p for label, p in sorted_vertices[:4]]
        
        for p in points_list:
            coords.extend([p[0], p[1]])
            
        # Pad by REPLICATING the last point (P4=P3)
        if points_list:
            last_p = points_list[-1]
            while len(coords) < 8:
                coords.extend([last_p[0], last_p[1]])
        else:
            coords = [0.0] * 8
            
        # Normalize: / 20.0
        input_tensor = torch.tensor(coords, dtype=torch.float32) / 20.0
        
        # --- TARGET ---
        # Calculate all properties from the sorted coordinates
        # We need to compute [Angle1, Angle2, Angle3, Side12, Side23, Side31]
        # P1, P2, P3 are the first 3 points.
        
        # P4 might exist, but primary triangle is P1-P2-P3.
        # This assumes the main shape is a triangle (true for our dataset).
        
        import math
        
        def dist(a, b):
            return math.hypot(a[0]-b[0], a[1]-b[1])
            
        def get_angle(a, b, c):
            # Angle at b (between ba and bc)
            ba = (a[0]-b[0], a[1]-b[1])
            bc = (c[0]-b[0], c[1]-b[1])
            dot = ba[0]*bc[0] + ba[1]*bc[1]
            cross = ba[0]*bc[1] - ba[1]*bc[0]
            angle = math.degrees(math.atan2(cross, dot))
            return abs(angle) # Interior angle usually
            
        # Get points from sorted list for consistency with input
        p1 = points_list[0] if len(points_list) > 0 else [0,0]
        p2 = points_list[1] if len(points_list) > 1 else [0,0]
        p3 = points_list[2] if len(points_list) > 2 else [0,0]
        
        s12 = dist(p1, p2)
        s23 = dist(p2, p3)
        s31 = dist(p3, p1)
        
        a1 = get_angle(p3, p1, p2) # Angle at P1
        a2 = get_angle(p1, p2, p3) # Angle at P2
        a3 = get_angle(p2, p3, p1) # Angle at P3
        
        # Target Vector: [A1, A2, A3, S12, S23, S31]
        # Normalize?
        # Angles / 180.0
        # Sides / 20.0
        target_list = [
            a1 / 180.0,
            a2 / 180.0,
            a3 / 180.0,
            s12 / 20.0,
            s23 / 20.0,
            s31 / 20.0
        ]
        
        target_tensor = torch.tensor(target_list, dtype=torch.float32)
        
        # Input: (coords, type_idx)
        return (input_tensor, torch.tensor(type_idx, dtype=torch.long)), target_tensor

def get_solver_dataloaders(data_dir, batch_size=64, num_workers=4):
    train_ds = SolverDataset(data_dir, split="train")
    val_ds = SolverDataset(data_dir, split="val")
    # test_ds = SolverDataset(data_dir, split="test") # Optional
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dl, val_dl
