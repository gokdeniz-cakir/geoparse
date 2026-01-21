"""
Dataset for Keypoint Detection Training.

Generates Gaussian heatmap targets from ground truth vertex coordinates.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np

from .keypoint_model import generate_heatmap


class KeypointDataset(Dataset):
    """
    Dataset for keypoint detection training.
    
    Returns:
        image: Preprocessed image tensor [3, 224, 224]
        target_heatmap: Gaussian heatmap [4, 56, 56]
        coords: Normalized coordinates [4, 2] in range [0, 1]
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        heatmap_size: int = 56,
        sigma: float = 2.0,
        image_size: int = 224,
        coord_scale: float = 20.0
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.image_size = image_size
        self.coord_scale = coord_scale
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.jsonl"
        self.samples = []
        with open(metadata_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        # Split data (80/10/10)
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(0.8 * n)]
        elif split == "val":
            self.samples = self.samples[int(0.8 * n):int(0.9 * n)]
        else:  # test
            self.samples = self.samples[int(0.9 * n):]
            
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load image
        img_path = self.data_dir / "images" / item["filename"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Extract coordinates from structure
        structure = item["structure"]
        vertices = structure.get("vertices", {})
        
        # Get coordinates in consistent order
        # Structure format: {"A": [x, y], "B": [x, y], "C": [x, y], ...}
        # Sort by label to ensure consistent ordering
        sorted_labels = sorted(vertices.keys())
        
        coords_list = []
        for label in sorted_labels[:4]:  # Max 4 vertices
            if label in vertices:
                coords_list.append(vertices[label])
                
        # Pad to 4 vertices if needed (repeat last)
        while len(coords_list) < 4:
            coords_list.append(coords_list[-1] if coords_list else [0, 0])
            
        coords = np.array(coords_list, dtype=np.float32)  # [4, 2]
        
        # Convert from geometry space [-scale, scale] to normalized [0, 1]
        # x_norm = (x + scale) / (2 * scale)
        coords_norm = (coords + self.coord_scale) / (2 * self.coord_scale)
        
        # Clip to valid range
        coords_norm = np.clip(coords_norm, 0, 1)
        
        # Convert to tensor
        coords_tensor = torch.from_numpy(coords_norm)
        
        # Generate target heatmap
        target_heatmap = generate_heatmap(
            coords_tensor, 
            heatmap_size=self.heatmap_size,
            sigma=self.sigma
        )
        
        return {
            'image': image,
            'heatmap': target_heatmap,
            'coords': coords_tensor,
            'filename': item['filename']
        }


def visualize_heatmaps(dataset, idx=0, save_path="heatmap_vis.png"):
    """Visualize sample heatmaps for debugging."""
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    image = sample['image']
    heatmap = sample['heatmap']
    coords = sample['coords']
    
    # Denormalize image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = image * std + mean
    img_display = img_display.permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")
    
    # Individual heatmaps
    for i in range(4):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]
        ax.imshow(heatmap[i].numpy(), cmap='hot')
        ax.set_title(f"P{i+1} Heatmap")
        ax.axis("off")
        
    # Combined heatmap
    combined = heatmap.max(dim=0)[0]
    axes[1, 2].imshow(combined.numpy(), cmap='hot')
    axes[1, 2].set_title("Combined Heatmap")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    

if __name__ == "__main__":
    # Test dataset
    dataset = KeypointDataset(
        data_dir="data/dataset",
        split="train",
        heatmap_size=56,
        sigma=2.0
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Heatmap shape: {sample['heatmap'].shape}")
    print(f"Coords shape: {sample['coords'].shape}")
    print(f"Coords: {sample['coords']}")
    
    # Visualize
    visualize_heatmaps(dataset, idx=0, save_path="heatmap_vis.png")
    print("✅ KeypointDataset test passed!")
