"""
Vision model for diagram parsing.
"""

import torch
import torch.nn as nn
from torchvision import models


class DiagramParser(nn.Module):
    """
    Vision model that extracts structure from geometry diagrams.
    
    Architecture:
    - Pretrained ResNet18 backbone (frozen or fine-tuned)
    - MLP head for regression outputs
    
    Outputs:
    - 9 values: [Ax, Ay, Bx, By, Cx, Cy, AB, BC, CA] (normalized)
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
        output_dim: int = 8  # 4 vertices * 2 coords
    ):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            backbone_out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification head
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Image tensor of shape (B, 3, H, W)
            
        Returns:
            Tensor of shape (B, 8) with predicted structure values
        """
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def predict_structure(self, x):
        """
        Convert raw outputs back to structure format.
        
        Returns list of dicts with vertices.
        """
        self.eval()
        with torch.no_grad():
            output = self(x)
        
        # Denormalize
        output = output.cpu().numpy()
        
        results = []
        for i in range(output.shape[0]):
            pred = output[i]
            
            # Denormalize vertices (multiply by 20.0 to match dataset)
            # Output is [Ax, Ay, Bx, By, Cx, Cy, Dx, Dy]
            # Coordinates are now P1, P2, P3, P4 in sorted order
            
            p1 = [float(pred[0] * 20.0), float(pred[1] * 20.0)]
            p2 = [float(pred[2] * 20.0), float(pred[3] * 20.0)]
            p3 = [float(pred[4] * 20.0), float(pred[5] * 20.0)]
            p4 = [float(pred[6] * 20.0), float(pred[7] * 20.0)]
            
            vertices = {
                "P1": p1,
                "P2": p2,
                "P3": p3,
            }
            
            # Check if P4 is a distinct point from P3
            # Dataset pads by duplicating P3. So if dist(P4, P3) is small, ignore P4.
            dist_sq = (p4[0]-p3[0])**2 + (p4[1]-p3[1])**2
            if dist_sq > 0.5:  # Threshold (~0.7 units)
                 vertices["P4"] = p4
            
            results.append({
                "type": "geometry_diagram",
                "vertices": vertices
            })
        
        return results
