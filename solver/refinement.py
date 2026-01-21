
import torch
import torch.nn as nn
import torch.optim as optim

def refine_coordinates(coords_pred_norm, problem_type):
    """
    Refine coordinates using geometric constraints.
    Input: Normalized coordinates [1, 8]
    Output: Refined normalized coordinates
    """
    # Clone and enable gradients
    coords = coords_pred_norm.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([coords], lr=0.01)
    
    # Define distinct points
    def get_points(c):
        p1 = c[0, 0:2]
        p2 = c[0, 2:4]
        p3 = c[0, 4:6]
        return p1, p2, p3

    # Optimization Loop
    for i in range(100):
        optimizer.zero_grad()
        p1, p2, p3 = get_points(coords)
        
        # 1. Fidelity Loss (Stay close to prediction)
        loss_fidelity = torch.mean((coords - coords_pred_norm.detach())**2)
        
        # 2. Geometric Constraints
        loss_geo = 0.0
        
        # Constraint: Right Triangle (Angle at A=90?)
        # We need to know WHICH angle is 90. 
        # For our generator, 'RightTriangle' usually has C at 90, but rotation changes it.
        # Heuristic: Soft constraint on the angle closest to 90.
        
        v1 = p2 - p1 # P1->P2
        v2 = p3 - p1 # P1->P3
        v3 = p3 - p2 # P2->P3
        
        # Constraint: Global Angle Sum = 180 (Pi radians)
        # This applies to ALL triangles
        
        # Calculate angles in radians
        # Angle at P1 (between v1 and v2)
        def get_angle(u, v):
            dot = torch.sum(u * v)
            norm = torch.norm(u) * torch.norm(v) + 1e-6
            cos = torch.clamp(dot / norm, -1.0, 1.0)
            return torch.acos(cos)
            
        a1 = get_angle(v1, v2)
        a2 = get_angle(-v1, v3) # angle at P2 between P2->P1 and P2->P3
        a3 = get_angle(-v2, -v3) # angle at P3
        
        angle_sum = a1 + a2 + a3
        loss_angle_sum = (angle_sum - torch.pi)**2
        loss_geo += loss_angle_sum * 10.0 # High weight for validity
            
        if "right" in problem_type:
            # Minimize (angle - pi/2)^2 for the one closest to 90
            l1 = (a1 - torch.pi/2)**2
            l2 = (a2 - torch.pi/2)**2
            l3 = (a3 - torch.pi/2)**2
            loss_geo += min(l1, l2, l3) * 5.0

        if "isosceles" in problem_type:
            # Two sides should be equal. 
            # Minimize (S1-S2)^2 or (S2-S3)^2 or (S3-S1)^2
            d1 = torch.norm(v1)
            d2 = torch.norm(v2)
            d3 = torch.norm(v3)
            
            diff1 = (d1 - d2)**2
            diff2 = (d2 - d3)**2
            diff3 = (d3 - d1)**2
            
            loss_geo += min(diff1, diff2, diff3) * 5.0
            
        # Total Loss
        loss = loss_fidelity + loss_geo * 5.0 # Weight constraints higher
        loss.backward()
        optimizer.step()
        
    return coords.detach()
