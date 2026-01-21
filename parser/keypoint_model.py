"""
Keypoint Detection Model for Robust Geometry Parsing.

Architecture:
- Encoder: ResNet18 (pretrained) extracts feature maps
- Decoder: Upsampling + Conv layers produce heatmaps
- Output: 4-channel heatmap [B, 4, H, W] where each channel is P1-P4 probability

Why Heatmaps?
- Local features are style-invariant (corners look like corners)
- Translation equivariant (learns "what" not "where")
- Robust to domain shift (textbooks, whiteboards, sketches)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class KeypointDetector(nn.Module):
    """
    Heatmap-based keypoint detector for geometry diagrams.
    
    Input: [B, 3, 224, 224] RGB image
    Output: [B, 4, 56, 56] heatmaps for 4 vertices
    """
    
    def __init__(self, num_keypoints: int = 4, heatmap_size: int = 56):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        
        # Encoder: ResNet18 backbone (remove final FC layers)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Keep layers up to layer4 (output: [B, 512, 7, 7] for 224x224 input)
        self.encoder = nn.Sequential(
            resnet.conv1,      # -> [B, 64, 112, 112]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # -> [B, 64, 56, 56]
            resnet.layer1,     # -> [B, 64, 56, 56]
            resnet.layer2,     # -> [B, 128, 28, 28]
            resnet.layer3,     # -> [B, 256, 14, 14]
            resnet.layer4,     # -> [B, 512, 7, 7]
        )
        
        # Decoder: Upsample back to heatmap resolution
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final conv to keypoint channels
            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, 224, 224]
            
        Returns:
            Heatmaps [B, num_keypoints, heatmap_size, heatmap_size]
        """
        features = self.encoder(x)
        heatmaps = self.decoder(features)
        return heatmaps
    
    def predict_keypoints(self, x: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
        """
        Predict keypoint coordinates from image.
        
        Args:
            x: Input image [B, 3, 224, 224]
            temperature: Scaling factor for softmax sharpening (higher = sharper peaks)
            
        Returns:
            Coordinates [B, num_keypoints, 2] in range [0, 1] (normalized)
        """
        heatmaps = self.forward(x)  # [B, K, H, W]
        B, K, H, W = heatmaps.shape
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, K, -1)  # [B, K, H*W]
        
        # Apply temperature scaling to sharpen peaks before softmax
        softmax = F.softmax(heatmaps_flat * temperature, dim=-1)  # [B, K, H*W]
        
        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=x.device)
        x_coords = torch.linspace(0, 1, W, device=x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        xx = xx.flatten()  # [H*W]
        yy = yy.flatten()  # [H*W]
        
        # Weighted sum of coordinates
        pred_x = (softmax * xx.view(1, 1, -1)).sum(dim=-1)  # [B, K]
        pred_y = (softmax * yy.view(1, 1, -1)).sum(dim=-1)  # [B, K]
        
        # Stack to [B, K, 2]
        coords = torch.stack([pred_x, pred_y], dim=-1)
        
        return coords
    
    def predict_structure(self, image: torch.Tensor, scale: float = 20.0) -> dict:
        """
        Predict geometry structure from image (for inference).
        
        Args:
            image: Preprocessed image tensor [1, 3, 224, 224]
            scale: Denormalization scale for coordinates
            
        Returns:
            Dictionary with vertices and metadata
        """
        self.eval()
        with torch.no_grad():
            coords = self.predict_keypoints(image)  # [1, K, 2]
            
        coords = coords[0].cpu().numpy()  # [K, 2]
        
        # Convert from [0,1] to geometry space [-scale, scale]
        coords = (coords - 0.5) * 2 * scale
        
        # Detect number of valid vertices (P4 close to P3 means 3-vertex shape)
        num_vertices = 4
        if len(coords) >= 4:
            dist_p3_p4 = ((coords[2] - coords[3]) ** 2).sum() ** 0.5
            if dist_p3_p4 < 1.0:
                num_vertices = 3
        
        vertices = {}
        labels = ['P1', 'P2', 'P3', 'P4']
        for i in range(num_vertices):
            vertices[labels[i]] = coords[i].tolist()
            
        return {
            'type': 'triangle' if num_vertices == 3 else 'quadrilateral',
            'vertices': vertices,
            'num_vertices': num_vertices
        }


def generate_heatmap(
    coords: torch.Tensor, 
    heatmap_size: int = 56,
    sigma: float = 2.0
) -> torch.Tensor:
    """
    Generate Gaussian heatmap targets for training.
    
    Args:
        coords: Normalized coordinates [K, 2] in range [0, 1]
        heatmap_size: Size of output heatmap
        sigma: Gaussian standard deviation in pixels
        
    Returns:
        Heatmap [K, heatmap_size, heatmap_size]
    """
    K = coords.shape[0]
    device = coords.device
    
    # Create coordinate grid
    y = torch.arange(heatmap_size, device=device).float()
    x = torch.arange(heatmap_size, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Scale coordinates to heatmap space
    cx = coords[:, 0] * heatmap_size  # [K]
    cy = coords[:, 1] * heatmap_size  # [K]
    
    # Expand for broadcasting: [K, H, W]
    cx = cx.view(K, 1, 1)
    cy = cy.view(K, 1, 1)
    
    # Gaussian blob
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    heatmap = torch.exp(-dist_sq / (2 * sigma ** 2))
    
    return heatmap


if __name__ == "__main__":
    # Quick test
    model = KeypointDetector(num_keypoints=4, heatmap_size=56)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    heatmaps = model(dummy_input)
    print(f"Heatmap shape: {heatmaps.shape}")  # Expected: [2, 4, 56, 56]
    
    # Test coordinate prediction
    coords = model.predict_keypoints(dummy_input)
    print(f"Coords shape: {coords.shape}")  # Expected: [2, 4, 2]
    
    # Test heatmap generation
    test_coords = torch.tensor([[0.2, 0.3], [0.8, 0.2], [0.5, 0.9], [0.5, 0.9]])
    target_heatmap = generate_heatmap(test_coords)
    print(f"Target heatmap shape: {target_heatmap.shape}")  # Expected: [4, 56, 56]
    
    print("✅ KeypointDetector test passed!")
