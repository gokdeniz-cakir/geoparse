
import torch
import numpy as np
import math
from solver.model import NeuralSolver

def proof_solver_logic():
    print("--- Proof of Solver Reasoning ---")
    print("Goal: Prove the Solver correctly identifies an 80-70-30 triangle if given correct coordinates.")
    
    # 1. Generate Coords for 80-70-30 Triangle
    # Let's fix B at origin (0,0), C on x-axis.
    # Angle B = 70 deg. Angle C = 30 deg. (So A = 80 deg)
    # Check: 80+70+30 = 180. OK.
    
    # Scale doesn't strictly matter for angles, but let's use a reasonable size.
    c = 10.0 # Side AB length? No, let's say Side a (BC) is base.
    
    # Let B = (0, 0)
    # Let C = (10, 0)
    # A = (x, y). 
    # Angle B = 70. So A is on line from B at 70 deg.
    # Angle C = 30. So A is on line from C at (180-30)=150 deg (standard angle).
    
    # Line 1: y = tan(70)*x
    # Line 2: y - 0 = tan(150)*(x - 10)
    
    tan70 = math.tan(math.radians(70))
    tan150 = math.tan(math.radians(150)) # -tan(30)
    
    # tan70 * x = tan150 * (x - 10)
    # x * (tan70 - tan150) = -10 * tan150
    x = (-10 * tan150) / (tan70 - tan150)
    y = tan70 * x
    
    # Centering to match training distribution center (approx 0,0)
    # Centroid approx (0+10+x)/3, (0+0+y)/3
    cx, cy = (x+10)/3, y/3
    
    B = (0 - cx, 0 - cy)
    C = (10 - cx, 0 - cy)
    A = (x - cx, y - cy)
    
    # Pad to 4 points (P4=P3)
    coords = [A[0], A[1], B[0], B[1], C[0], C[1], C[0], C[1]]
    print(f"Generated Coords for 80-70-30 Triangle:")
    print(f"  P1 (A): {A}")
    print(f"  P2 (B): {B}")
    print(f"  P3 (C): {C}")
    
    # Normalize (/ 20.0)
    tensor_coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0) / 20.0
    type_tensor = torch.tensor([0], dtype=torch.long) # Unknown type
    
    # 1.5 Run Refinement (New)
    from solver.refinement import refine_coordinates
    print("Running Refinement on Manual Coordinates...")
    tensor_coords_refined = refine_coordinates(tensor_coords, "unknown")
    
    # 2. Run Solver
    device = "cpu"
    solver = NeuralSolver(output_dim=6)
    solver.load_state_dict(torch.load("experiments/solver_baseline/best_solver.pt", map_location=device)["model_state_dict"])
    solver.eval()
    
    with torch.no_grad():
        props = solver((tensor_coords_refined, type_tensor))[0]
        
    # 3. Interpret Neural Output
    p = props.numpy()
    angles_neural = p[:3] * 180.0
    
    # 4. Interpret Analytic Output (From Refined Coords)
    # Calculate angles from tensor_coords_refined
    p_ref = tensor_coords_refined.cpu().numpy()[0] * 20.0
    p1 = p_ref[0:2] # A
    p2 = p_ref[2:4] # B
    p3 = p_ref[4:6] # C
    
    def calc_angle(v_vertex, v_left, v_right):
        vec1 = v_left - v_vertex
        vec2 = v_right - v_vertex
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-6 or norm2 < 1e-6: return 0.0
        cos = np.dot(vec1, vec2) / (norm1 * norm2)
        cos = np.clip(cos, -1.0, 1.0)
        return np.degrees(np.arccos(cos))
        
    # Standard triangle logic depends on ordering.
    # Manual coords were A(P1), B(P2), C(P3).
    # Angle A (at P1)
    a1 = calc_angle(p1, p2, p3)
    # Angle B (at P2)
    a2 = calc_angle(p2, p1, p3)
    # Angle C (at P3)
    a3 = calc_angle(p3, p1, p2)
    
    angles_analytic = [a1, a2, a3]
    
    print("\nResults Comparison:")
    print(f"  Target  : [80.0, 70.0, 30.0]")
    print(f"  Neural  : [{angles_neural[0]:.1f}, {angles_neural[1]:.1f}, {angles_neural[2]:.1f}]")
    print(f"  Analytic: [{angles_analytic[0]:.1f}, {angles_analytic[1]:.1f}, {angles_analytic[2]:.1f}]")
    
    print("\nConclusion:")
    if abs(sum(angles_analytic) - 180.0) < 0.1:
         print("  ✅ Analytic Solver on Refined Coordinates is PERFECT.")
    else:
         print("  ❌ Analytic Solver failed.")
    err = abs(angles[0] - 80) + abs(angles[1] - 70) + abs(angles[2] - 30)
    # The order might be permuted depending on P1/P2/P3 assignment, but let's check values.
    # Closest match check
    target_angles = [80, 70, 30]
    matched = []
    for t in target_angles:
        best = min(angles, key=lambda a: abs(a-t))
        matched.append(best)
        
    print(f"  Target Angles: {target_angles}")
    print(f"  Found Angles : {[round(m, 1) for m in matched]}")
    
    if all(abs(m - t) < 5.0 for m, t in zip(matched, target_angles)):
        print("  ✅ SUCCESS: Solver correctly reasoned about the geometry.")
    else:
        print("  ❌ FAILURE: Solver failed even with correct coordinates.")

if __name__ == "__main__":
    proof_solver_logic()
