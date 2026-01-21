"""
PyTorch dataset for geometry diagram parsing.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DiagramDataset(Dataset):
    """
    Dataset of (image, structure) pairs for training the parser.
    
    Outputs:
    - image: Tensor of shape (3, H, W)
    - target: Dict with normalized vertex coordinates and side lengths
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
        max_samples: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        
        # Load metadata from JSONL
        metadata_path = self.data_dir / "metadata.jsonl"
        self.metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))
            
        # Optional: limit total samples before splitting
        if max_samples is not None:
            self.metadata = self.metadata[:max_samples]
        
        # Simple split: 80% train, 10% val, 10% test
        n = len(self.metadata)
        if split == "train":
            self.metadata = self.metadata[:int(0.8 * n)]
        elif split == "val":
            self.metadata = self.metadata[int(0.8 * n):int(0.9 * n)]
        elif split == "test":
            self.metadata = self.metadata[int(0.9 * n):]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # Standard ImageNet normalization
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        # New dataset has filename in "filename", relative to data_dir/images
        image_path = self.data_dir / "images" / item["filename"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Extract structure
        structure = item["structure"]
        vertices = structure["vertices"]
        
        # We need to target a fixed size tensor. 
        # Support up to 4 points: A, B, C, D
        # Output: [Ax, Ay, Bx, By, Cx, Cy, Dx, Dy]
        # Normalize by 10 (assuming max coord is around 10)
        
        coords = []
        for label in ["A", "B", "C", "D"]:
            if label in vertices:
                # Assuming simple dict {"A": [x, y]} or keys that map to it
                # The randomization might have changed keys to "K", "L", etc.
                # But the structure["vertices"] dict has the randomized keys.
                # Wait, if keys are random, how do we order them?
                # We need a stable ordering. 
                # If structure has randomized keys, we can't look up "A".
                
                # Check if we have standard keys or need to sort
                pass
        
        v_keys = sorted(vertices.keys())
        
        # Canonical ordering: Sort by position to ensure consistent model targets
        # Primary: Top-most (max Y) -> represented as -y for ascending sort
        # Secondary: Left-most (min X) -> represented as x
        sorted_vertices = sorted(
            vertices.items(), 
            key=lambda v: (-v[1][1], v[1][0])
        )
        
        target_coords = []
        
        # Take up to 4 points
        points_list = [p for label, p in sorted_vertices[:4]]
        
        for p in points_list:
            target_coords.extend([p[0], p[1]])
            
        # Pad by REPLICATING the last point
        # This teaches the model: "If there is no 4th point, just stay at the 3rd point"
        # Heavily penalizes "guessing" a random location like (0,0)
        if points_list:
            last_p = points_list[-1]
            while len(target_coords) < 8:
                target_coords.extend([last_p[0], last_p[1]])
        else:
            # Fallback for empty (shouldn't happen)
            target_coords = [0.0] * 8
            
        # Normalize by 20.0 to cover range [-13, 14] securely
        target = torch.tensor(target_coords, dtype=torch.float32) / 20.0
        
        return image, target, item["answer"]


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    max_samples: Optional[int] = None
):
    """Create train, val, test dataloaders."""
    from torch.utils.data import DataLoader
    
    train_dataset = DiagramDataset(data_dir, split="train", image_size=image_size, max_samples=max_samples)
    val_dataset = DiagramDataset(data_dir, split="val", image_size=image_size, max_samples=max_samples)
    test_dataset = DiagramDataset(data_dir, split="test", image_size=image_size, max_samples=max_samples)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
